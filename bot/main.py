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


def _parse_ffprobe_fps(fps_str: str) -> float:
    """Parse FPS string from ffprobe (e.g., '30/1' or '29.97')."""
    try:
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            return num / den if den != 0 else 15.0
        return float(fps_str)
    except (ValueError, ZeroDivisionError):
        return 15.0  # A sensible default


def _get_video_metadata(video_path: Path) -> dict | None:
    """Run ffprobe to get video metadata."""
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        logger.warning('ffprobe not found, cannot get video metadata.')
        return None

    cmd = [
        ffprobe,
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate,duration',
        '-of', 'csv=p=0',
        str(video_path)
    ]
    try:
        # Use synchronous subprocess.run consistent with other calls
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=15)
        output = result.stdout.strip()

        width, height, fps_str, duration_str = output.split(',')

        return {
            'width': int(width),
            'height': int(height),
            'fps': _parse_ffprobe_fps(fps_str),
            'duration': float(duration_str)
        }
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
        logger.exception(f"Failed to get video metadata for {video_path}: {e}")
        return None


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

        try:
            logger.info('Starting conversion for chat=%s message_id=%s', msg.chat_id, msg.message_id)
            await msg.reply_text('Analyzing video properties...')

            # Get video metadata to make smarter conversion decisions
            metadata = _get_video_metadata(infile_path)
            if not metadata:
                await msg.reply_text('Failed to read video metadata. Cannot convert.')
                return

            # Smartly determine conversion parameters based on metadata
            duration = metadata['duration']
            fps = metadata['fps']
            width = metadata['width']
            height = metadata['height']

            # 1. Cap duration to a reasonable length for a GIF
            max_duration = 15.0
            if duration > max_duration:
                await retry_send_text_by_msg(msg, f"Video is {duration:.1f}s long; trimming to {max_duration}s for the GIF.")
                duration = max_duration
            else:
                max_duration = None  # Use full length if shorter than max

            # 2. Iteratively adjust FPS and width to meet a "pixel budget"
            # This is a heuristic to estimate output size before conversion.
            # A 480p, 15fps, 10s video is a good baseline (480*270*15*10 = ~20M "pixels")
            pixel_budget = 25_000_000
            pixel_score = width * height * fps * (duration or metadata['duration'])

            scale_width = width
            final_fps = min(fps, 30)  # Cap source FPS at 30

            # If the estimated "pixel score" is high, be more aggressive from the start.
            if pixel_score > pixel_budget and width > 320: # only if score is high and res is not already low
                await retry_send_text_by_msg(msg, "Video has high resolution/fps. Applying smart compression to create a smaller GIF...")
                
                # Reduce width and fps based on how much it's over budget
                reduction_factor = (pixel_score / pixel_budget) ** 0.5  # sqrt makes reduction less drastic
                
                scale_width = int(width / reduction_factor)
                final_fps = int(fps / reduction_factor)

                # Clamp to reasonable values for a GIF
                scale_width = max(min(scale_width, 480), 320)  # Clamp between 320p and 480p width
                final_fps = max(min(final_fps, 15), 10)  # Clamp between 10 and 15 fps
            else:
                # Looks good, use default high-quality settings
                scale_width = min(width, 480)  # Don't upscale, but cap at 480p for sanity
                final_fps = min(fps, 15)  # 15 fps is a good default for GIFs

            await retry_send_text_by_msg(msg, f'Converting to GIF ({scale_width}p width, {final_fps} fps)...')
            
            # Generate palette and then use it for better colors
            _run_ffmpeg_generate_palette(infile_path, palette, max_duration, final_fps, scale_width)
            logger.info('Palette generated: %s', palette)
            _run_ffmpeg_palette_use(infile_path, palette, out_gif, max_duration, final_fps, scale_width)
            
            initial_size = out_gif.stat().st_size
            logger.info('GIF generated: %s (size=%.1fKB)', out_gif.name, initial_size / 1024)

            # Check against Telegram's hard limit
            if initial_size > MAX_FILE_SIZE:
                await msg.reply_text(f'Generated GIF is too large to send ({initial_size / 1024 / 1024:.1f} MB). Try a shorter clip.')
                return

            # Compress if needed to get under the preferred conversational size
            if initial_size > MAX_GIF_SIZE:
                await retry_send_text_by_msg(msg, f'Initial GIF is {initial_size / 1024 / 1024:.1f}MB, running post-compression...')
            final_gif_path = await compress_gif(out_gif, max_size=MAX_GIF_SIZE)
            
            final_gif = Path(final_gif_path)
            final_size = final_gif.stat().st_size
            logger.info('Final GIF selected: %s size=%.1fKB', final_gif.name, final_size / 1024)

            if final_size > MAX_GIF_SIZE:
                await retry_send_text_by_msg(msg, f'Compressed GIF is still {final_size / 1024 / 1024:.1f}MB. Sending the smallest version we have.')
            else:
                await retry_send_text_by_msg(msg, f'Prepared GIF: {final_gif.name} ({final_size/1024:.1f} KB). Sending now...')

            # Send the resulting GIF
            await retry_send_animation(context.bot, msg.chat_id, str(final_gif))
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