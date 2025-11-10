import os
import logging
import tempfile
import shutil
import time
import subprocess
from pathlib import Path
import asyncio

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import TimedOut, NetworkError

# telegram.Request location varies between versions; try both known locations
try:
    from telegram.request import Request  # PTB v20+
except Exception:
    try:
        from telegram.utils.request import Request  # older PTB
    except Exception:
        Request = None
        # logger is defined later; create a temporary fallback logger to warn now
        import logging as _logging
        _logging.getLogger(__name__).warning('telegram.Request not importable; request timeouts will not be configured')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment
TOKEN = os.getenv('VIDEO2GIF_BOT_TOKEN')

if not TOKEN:
    logger.error('Environment variable VIDEO2GIF_BOT_TOKEN is not set')
    raise SystemExit('VIDEO2GIF_BOT_TOKEN required')

# Maximum file size for Telegram (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024
# Preferred maximum GIF size for conversations (can be tuned via env var, MB)
# Default raised per request to 5 MB
MAX_GIF_SIZE = int(os.getenv('VIDEO2GIF_MAX_GIF_SIZE_MB', '5')) * 1024 * 1024


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'Send me a video (or video file) and I will convert it to a GIF. Short clips work best.\n'
        'Note: the container must have ffmpeg installed.'
    )


def _check_ffmpeg():
    return shutil.which('ffmpeg') is not None


def _run_ffmpeg_generate_palette(input_path: Path, palette_path: Path, max_duration: int | None, fps: int, scale_width: int | None):
    vf_parts = [f'fps={fps}']
    if scale_width:
        vf_parts.append(f'scale={scale_width}:-1:flags=lanczos')
    vf = ','.join(vf_parts)
    cmd = ['ffmpeg', '-y', '-i', str(input_path)]
    if max_duration:
        cmd += ['-t', str(max_duration)]
    cmd += ['-vf', vf + ',palettegen', str(palette_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _run_ffmpeg_palette_use(input_path: Path, palette_path: Path, out_path: Path, max_duration: int | None, fps: int, scale_width: int | None):
    vf_parts = [f'fps={fps}']
    if scale_width:
        vf_parts.append(f'scale={scale_width}:-1:flags=lanczos')
    vf = ','.join(vf_parts)
    cmd = [
        'ffmpeg', '-y', '-i', str(input_path), '-i', str(palette_path)
    ]
    if max_duration:
        cmd += ['-t', str(max_duration)]
    cmd += ['-lavfi', vf + '[x];[x][1:v]paletteuse', str(out_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


async def compress_gif(input_path, max_size=MAX_GIF_SIZE):
    """Try several tools/strategies to reduce GIF size until it's <= max_size.

    Strategies tried (in order):
      1. gifsicle color reduction and optimization (fast, high quality)
      2. ImageMagick convert with resize + color reduction + optimization
      3. ffmpeg re-encode with progressively lower fps and width

    Returns path to the (possibly new) GIF to send. If no tool is available or
    compression can't reduce below max_size, returns original path.
    """
    inp = Path(input_path)
    try:
        size = inp.stat().st_size
    except Exception:
        logger.exception('Failed to stat input GIF %s', input_path)
        return str(input_path)

    if size <= max_size:
        return str(input_path)

    workdir = inp.parent
    gifsicle = shutil.which('gifsicle')
    convert = shutil.which('convert')

    # Total time budget for compression (seconds)
    time_budget = int(os.getenv('VIDEO2GIF_COMPRESS_TIME_SEC', '30'))
    start_time = time.monotonic()

    # split budget per stage to avoid earlier stages starving later ones
    # give at least a few seconds per stage
    per_stage = max(6, time_budget // 3)

    # helper to compute time left
    def time_left():
        return time_budget - (time.monotonic() - start_time)

    # 1) gifsicle: do color-preserving optimization first (fast, non-destructive)
    #    Avoid aggressive color reduction (--colors) unless absolutely necessary
    if gifsicle:
        logger.info('Using gifsicle for compression (non-destructive optimize, then conservative lossy)')
        try:
            last = inp
            # if little time left, skip gifsicle
            if time_left() < 2:
                logger.debug('Skipping gifsicle stage; not enough time left')
            else:
                # First try a non-color-changing optimize pass only
                out = workdir / f'{inp.stem}_gfs_opt.gif'
                cmd = [gifsicle, '--optimize=3', str(last), '-o', str(out)]
                try:
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                    if out.exists():
                        logger.info('gifsicle optimized %s size=%.1fKB', out.name, out.stat().st_size/1024)
                        if out.stat().st_size <= max_size:
                            logger.info('gifsicle optimize success: %s <= %d bytes', out, max_size)
                            return str(out)
                        last = out
                except subprocess.TimeoutExpired:
                    logger.debug('gifsicle optimize timeout')
                except subprocess.CalledProcessError:
                    logger.debug('gifsicle optimize attempt failed')

                # If non-destructive optimize didn't help enough, fall back to conservative lossy
                # Use conservative (small) lossy values to avoid severe quality loss / palette shifts
                for lossy in (30, 20):
                    if time_left() < 2:
                        logger.warning('Compression time budget low, breaking lossy stage')
                        break

                    out = workdir / f'{inp.stem}_gfs_lossy_{lossy}.gif'
                    cmd = [gifsicle, '--optimize=3', f'--lossy={lossy}', str(last), '-o', str(out)]
                    try:
                        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.debug('gifsicle lossy timeout for %s', out)
                        continue
                    except subprocess.CalledProcessError:
                        logger.debug('gifsicle lossy attempt failed (possibly unsupported)')
                        break

                    if out.exists():
                        logger.info('gifsicle produced %s size=%.1fKB', out.name, out.stat().st_size/1024)
                    if out.exists() and out.stat().st_size <= max_size:
                        logger.info('gifsicle lossy success: %s <= %d bytes', out, max_size)
                        return str(out)
                    last = out

        except Exception:
            logger.exception('gifsicle compression stage failed')

    # 2) ImageMagick convert (just resize and optimize, preserve original colors)
    if convert:
        logger.info('Using ImageMagick convert for resizing')
        try:
            # Skip ImageMagick if almost no time left
            if time_left() < 4:
                logger.debug('Skipping ImageMagick stage; not enough time left')
            else:
                # Only use ImageMagick for resizing, keep original colors
                for scale_pct in (75, 50):
                    if time_left() < 3:
                        logger.warning('Compression time budget low, breaking ImageMagick stage')
                        break

                    out = workdir / f'{inp.stem}_magick_{scale_pct}.gif'
                    # Simple resize pipeline:
                    # 1. Coalesce frames
                    # 2. Just resize with Lanczos
                    # 3. Optimize layers
                    cmd = [
                        convert, str(inp), '-coalesce',
                        '-filter', 'Lanczos',
                        '-resize', f'{scale_pct}%',
                        '-layers', 'Optimize',
                        str(out)
                    ]

                    try:
                        # Give more time for quality color processing
                        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.debug('ImageMagick timeout for scale=%d', scale_pct)
                        continue
                    except subprocess.CalledProcessError:
                        logger.debug('ImageMagick failed for scale=%d', scale_pct)
                        continue
                    
                    if out.exists():
                        logger.info('ImageMagick produced %s size=%.1fKB', out.name, out.stat().st_size/1024)
                        if out.stat().st_size <= max_size:
                            logger.info('ImageMagick success: %s <= %d bytes', out, max_size)
                            return str(out)

        except subprocess.CalledProcessError:
            logger.exception('ImageMagick stage failed')

    # 3) Re-encode with ffmpeg lower fps/width using palette method
    try:
        logger.info('Trying ffmpeg re-encode compression')
        # Prefer to keep more frames (higher fps) and only slowly reduce resolution.
        # This makes the final GIF slightly larger but visually smoother.
        for fps in (15, 14, 12, 10, 8):
            for width in (480, 420, 360, 320, 240):
                if time_left() < 4:
                    logger.warning('Compression time budget low, breaking ffmpeg stage')
                    break

                pal = workdir / f'{inp.stem}_pal_{fps}_{width}.png'
                candidate = workdir / f'{inp.stem}_ff_{fps}_{width}.gif'

                try:
                    logger.debug('ffmpeg palette gen fps=%d width=%s', fps, width)
                    # run palette generation with a timeout
                    try:
                        _run_ffmpeg_generate_palette(inp, pal, None, fps, width)
                    except Exception as e:
                        logger.debug('ffmpeg palette generation failed: %s', e)
                        continue

                    try:
                        _run_ffmpeg_palette_use(inp, pal, candidate, None, fps, width)
                    except Exception as e:
                        logger.debug('ffmpeg palette use failed: %s', e)
                        continue

                except Exception as ex:
                    logger.debug('ffmpeg attempt failed for fps=%d width=%s: %s', fps, width, ex)
                    continue

                if candidate.exists():
                    logger.info('ffmpeg produced %s size=%.1fKB', candidate.name, candidate.stat().st_size/1024)
                    if candidate.stat().st_size <= max_size:
                        logger.info('ffmpeg success: %s <= %d bytes', candidate, max_size)
                        return str(candidate)

    except Exception as e:
        logger.exception('ffmpeg compression loop failed: %s', e)

    # Nothing worked; return original GIF
    return str(input_path)


async def retry_send_animation(bot, chat_id, animation_path, max_retries=3):
    """Retry sending animation with exponential backoff"""
    for attempt in range(max_retries):
        try:
            logger.info('Sending animation attempt %d: %s', attempt+1, animation_path)
            with open(animation_path, 'rb') as animation:
                await bot.send_animation(chat_id=chat_id, animation=animation)
            logger.info('send_animation call returned for attempt %d', attempt+1)
            return True
        except (TimedOut, NetworkError) as e:
            logger.warning('send_animation attempt %d failed: %s', attempt+1, e)
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
    return False


async def retry_send_text_by_msg(msg, text, max_retries: int | None = None, **kwargs):
    """Retry sending a text reply via a Message object's reply_text with exponential backoff.

    Returns True on success, False on final failure.
    """
    if max_retries is None:
        max_retries = int(os.getenv('VIDEO2GIF_MESSAGE_RETRIES', '3'))
    for attempt in range(1, max_retries + 1):
        try:
            await msg.reply_text(text, **kwargs)
            return True
        except (TimedOut, NetworkError) as e:
            logger.warning('reply_text attempt %d/%d failed: %s', attempt, max_retries, e)
            if attempt == max_retries:
                logger.exception('Final reply_text attempt failed')
                return False
            await asyncio.sleep(2 ** (attempt - 1))
        except Exception as e:
            logger.exception('Unexpected error while sending reply_text: %s', e)
            return False


async def retry_send_message(bot, chat_id, text, max_retries: int | None = None, **kwargs):
    """Retry sending a message via bot.send_message with exponential backoff."""
    if max_retries is None:
        max_retries = int(os.getenv('VIDEO2GIF_MESSAGE_RETRIES', '3'))
    for attempt in range(1, max_retries + 1):
        try:
            await bot.send_message(chat_id=chat_id, text=text, **kwargs)
            return True
        except (TimedOut, NetworkError) as e:
            logger.warning('send_message attempt %d/%d failed: %s', attempt, max_retries, e)
            if attempt == max_retries:
                logger.exception('Final send_message attempt failed')
                return False
            await asyncio.sleep(2 ** (attempt - 1))
        except Exception as e:
            logger.exception('Unexpected error while sending message: %s', e)
            return False


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message

    # Determine file_id for incoming media
    file_obj = None
    if msg.video:
        file_obj = msg.video.get_file()
    elif msg.document and (msg.document.mime_type or '').startswith('video'):
        file_obj = msg.document.get_file()
    elif msg.animation:
        # Already a gif/animation — just inform user
        await msg.reply_text('This is already an animation/gif. I will resend it back to you.')
        # forward original animation
        await context.bot.send_animation(chat_id=msg.chat_id, animation=msg.animation.file_id)
        return
    else:
        await msg.reply_text('Please send a video file (mp4, mov, webm, etc.) or a video document.')
        return

    await msg.reply_text('Downloading file...')

    if not _check_ffmpeg():
        await msg.reply_text('ffmpeg is not installed in this environment. Please install ffmpeg in the container.')
        return

    # Create temporary working directory
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_path = td_path / 'input'
        out_gif = td_path / 'out.gif'
        palette = td_path / 'palette.png'

        # Download file with retries and backoff to handle transient network/timeouts
        f = await file_obj
        infile_path = in_path.with_suffix(Path(f.file_path).suffix or '.mp4')
        download_attempts = int(os.getenv('VIDEO2GIF_DOWNLOAD_RETRIES', '3'))
        for attempt in range(1, download_attempts + 1):
            try:
                await f.download_to_drive(custom_path=str(infile_path))
                break
            except (TimedOut, NetworkError) as e:
                logger.warning('Download attempt %d/%d failed: %s', attempt, download_attempts, e)
                if attempt == download_attempts:
                    await msg.reply_text('Failed to download file due to network timeout. Please try again later.')
                    return
                # exponential backoff
                await asyncio.sleep(2 ** (attempt - 1))
            except Exception as e:
                logger.exception('Unexpected error while downloading file: %s', e)
                await msg.reply_text('Failed to download the file. Please try again later.')
                return

        # Conversion parameters (reasonable defaults)
        # If input is large, be aggressive up-front to avoid creating very large GIFs
        infile_size = None
        try:
            infile_size = infile_path.stat().st_size
        except Exception:
            infile_size = None

        # default
        max_duration = 15  # seconds, trim longer videos
        fps = 15
        scale_width = 480  # scale width, keep aspect ratio

        # If the uploaded video is big, start with smaller defaults to keep gif size manageable
        if infile_size and infile_size > (8 * 1024 * 1024):  # >8MB
            max_duration = min(max_duration, 12)
            fps = 12
            scale_width = 360
        elif infile_size and infile_size > (20 * 1024 * 1024):
            # very large uploads: be more aggressive
            max_duration = min(max_duration, 10)
            fps = 10
            scale_width = 320

        try:
            logger.info('Starting conversion for chat=%s message_id=%s', msg.chat_id, msg.message_id)
            await msg.reply_text('Converting to GIF (this may take a few seconds)...')
            
            # Generate palette and then use it for better colors (parameters chosen above)
            _run_ffmpeg_generate_palette(infile_path, palette, max_duration, fps, scale_width)
            logger.info('Palette generated: %s', palette)
            _run_ffmpeg_palette_use(infile_path, palette, out_gif, max_duration, fps, scale_width)
            logger.info('GIF generated: %s (size=%.1fKB)', out_gif.name, out_gif.stat().st_size/1024)
            # Check size (Telegram bots have file size limits - commonly 50 MB)
            max_send_size = 50 * 1024 * 1024
            size = out_gif.stat().st_size
            if size > max_send_size:
                await msg.reply_text(f'GIF is too large to send ({size/1024/1024:.1f} MB). Try a shorter clip.')
                return

            # Compress if needed
            final_gif = await compress_gif(out_gif)
            try:
                final_size = Path(final_gif).stat().st_size
                logger.info('Final GIF selected: %s size=%.1fKB', Path(final_gif).name, final_size/1024)
                # Inform user about final size before sending
                try:
                    sent = await retry_send_text_by_msg(msg, f'Prepared GIF: {Path(final_gif).name} ({final_size/1024:.1f} KB). Sending now...')
                    if not sent:
                        logger.warning('Could not deliver prepared-GIF progress message after retries')
                except Exception:
                    logger.exception('Failed to send progress message to user')
            except Exception:
                logger.exception('Failed to stat final GIF %s', final_gif)

            # If the final GIF is still larger than our preferred GIF size, try one
            # more aggressive GIF re-encode (lower fps and width) and re-run compression.
            try:
                final_size = Path(final_gif).stat().st_size
            except Exception:
                final_size = None

            if final_size and final_size > MAX_GIF_SIZE:
                logger.info('Final GIF %s is larger than preferred max (%d bytes). Trying aggressive re-encode.', final_gif, MAX_GIF_SIZE)
                await msg.reply_text('GIF is still large after compression — trying a more aggressive small GIF.')
                
                # aggressive parameters (keep more fps/quality by default than before)
                # Raise defaults to preserve smoother motion; users can still override via env.
                aggressive_fps = int(os.getenv('VIDEO2GIF_AGGRESSIVE_FPS', '10'))
                aggressive_width = int(os.getenv('VIDEO2GIF_AGGRESSIVE_WIDTH', '320'))
                aggressive_gif = td_path / 'out_aggressive.gif'
                
                try:
                    # Recreate palette + aggressive gif
                    _run_ffmpeg_generate_palette(infile_path, palette, max_duration, aggressive_fps, aggressive_width)
                    _run_ffmpeg_palette_use(infile_path, palette, aggressive_gif, max_duration, aggressive_fps, aggressive_width)
                    logger.info('Aggressive GIF generated: %s (size=%.1fKB)', aggressive_gif.name, aggressive_gif.stat().st_size/1024)

                    # Try compressing the aggressive GIF as well
                    final_gif_candidate = await compress_gif(aggressive_gif)
                    try:
                        final_candidate_size = Path(final_gif_candidate).stat().st_size
                        logger.info('Final aggressive GIF selected: %s size=%.1fKB', Path(final_gif_candidate).name, final_candidate_size/1024)
                        sent = await retry_send_text_by_msg(msg, f'Prepared small GIF: {Path(final_gif_candidate).name} ({final_candidate_size/1024:.1f} KB). Sending now...')
                        if not sent:
                            logger.warning('Could not deliver aggressive prepared-GIF progress message after retries')
                    except Exception:
                        logger.exception('Failed to stat aggressive GIF %s', final_gif_candidate)
                    
                    # send whichever candidate we got
                    await retry_send_animation(context.bot, msg.chat_id, final_gif_candidate)
                    await msg.reply_text('Done!')

                except subprocess.CalledProcessError as e:
                    logger.exception('ffmpeg aggressive re-encode failed: %s', e)
                    await msg.reply_text('Failed to produce a smaller GIF after aggressive attempts. Sending the best GIF we have...')
                    await retry_send_animation(context.bot, msg.chat_id, final_gif)
                    await msg.reply_text('Done!')
                except Exception as e:
                    logger.exception('Unexpected error during aggressive GIF generation: %s', e)
                    await msg.reply_text('Unexpected error while producing smaller GIF. Sending the best GIF we have...')
                    await retry_send_animation(context.bot, msg.chat_id, final_gif)
                    await msg.reply_text('Done!')
            else:
                # Send the resulting GIF
                await retry_send_animation(context.bot, msg.chat_id, final_gif)
                await msg.reply_text('Done!')

        except subprocess.CalledProcessError as e:
            logger.exception('ffmpeg failed: %s', e)
            await msg.reply_text('Conversion failed. ffmpeg error.')
        except Exception as e:
            logger.exception('Unexpected error: %s', e)
            await msg.reply_text('An unexpected error occurred during conversion.')


async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Global error handler for the Application.

    Logs the exception and (when possible) notifies the user that something went wrong.
    """
    try:
        logger.exception('Unhandled exception while processing an update: %s', context.error)
    except Exception:
        # Fallback logging if something goes wrong while logging
        logger.exception('Unhandled exception (and failed to format context.error)')

    # Try to notify the user in the chat where the error happened
    try:
        if update and getattr(update, 'effective_chat', None):
            await context.bot.send_message(chat_id=update.effective_chat.id, text='An internal error occurred. Please try again later.')
    except Exception:
        logger.exception('Failed to notify user about the internal error')


def main():
    # Configure HTTP request timeouts for the underlying telegram/httpx client
    if Request is not None:
        req = Request(
            connect_timeout=float(os.getenv('VIDEO2GIF_CONNECT_TIMEOUT', '5')),
            read_timeout=float(os.getenv('VIDEO2GIF_READ_TIMEOUT', '30')),
            pool_timeout=float(os.getenv('VIDEO2GIF_POOL_TIMEOUT', '5')),
            con_pool_size=int(os.getenv('VIDEO2GIF_CONNECTION_POOL', '8')),
        )

        app = ApplicationBuilder().token(TOKEN).request(req).build()
    else:
        logger.warning('telegram.Request class unavailable; building Application without custom Request timeouts')
        app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler('start', start))
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO | filters.ANIMATION, handle_video))

    # Register a global error handler so uncaught exceptions are logged and optionally reported to users
    app.add_error_handler(global_error_handler)

    logger.info('Starting bot...')
    app.run_polling()


if __name__ == '__main__':
    main()