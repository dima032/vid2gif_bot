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
# Recommended maximum GIF size for savable GIFs (10MB)
MAX_GIF_SIZE = int(os.getenv('VIDEO2GIF_MAX_GIF_SIZE_MB', '10')) * 1024 * 1024


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        'Send me a video (or video file) and I will convert it to a silent mp4, so you can save it to your GIF tab. '
        'Short clips work best.\n'
        'Note: the container must have ffmpeg installed.'
    )


def _check_ffmpeg():
    return shutil.which('ffmpeg') is not None


def _run_ffmpeg_convert_mp4(input_path: Path, out_path: Path, fps: int, scale_width: int | None):
    # Use ceil(X/2)*2 to ensure dimensions are divisible by 2 for libx264
    # The pad filter adds black borders if necessary, but maintains aspect ratio.
    vf_parts = [f'fps={fps}']
    if scale_width:
        vf_parts.append(f'scale={scale_width}:-1:flags=lanczos')
    vf_parts.append('pad=ceil(iw/2)*2:ceil(ih/2)*2') # Ensure dimensions are even
    vf = ','.join(vf_parts)

    cmd = [
        'ffmpeg', '-y', '-i', str(input_path),
        '-vf', vf,
        '-an',  # No audio
        '-c:v', 'libx264',
        '-preset', 'veryfast',  # Faster encoding
        '-crf', '28', # Higher CRF for smaller file size, adjust as needed
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',  # Web optimization
        str(out_path)
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if result.stderr:
        logger.warning('ffmpeg stderr: %s', result.stderr)


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





async def retry_send_animation(bot, chat_id, animation_path, max_retries=3, duration: int | None = None, width: int | None = None, height: int | None = None):
    """Retry sending animation with exponential backoff"""
    for attempt in range(max_retries):
        try:
            logger.info('Sending animation attempt %d: %s', attempt+1, animation_path)
            with open(animation_path, 'rb') as animation:
                await bot.send_animation(
                    chat_id=chat_id,
                    animation=animation,
                    duration=duration,
                    width=width,
                    height=height,
                    write_timeout=120
                )
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
        out_mp4 = td_path / 'out.mp4'

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

            # max_duration = 15.0
            # if duration > max_duration:
            #     await retry_send_text_by_msg(msg, f"Video is {duration:.1f}s long; trimming to {max_duration}s.")
            #     duration = max_duration
            # else:
            #     max_duration = None  # Use full length if shorter than max


            # 2. Iteratively adjust FPS and width to meet a "pixel budget"
            # This is a heuristic to estimate output size before conversion.
            pixel_budget = 25_000_000
            pixel_score = width * height * fps * (duration or metadata['duration'])

            scale_width = width
            final_fps = min(fps, 30)  # Cap source FPS at 30

            # If the estimated "pixel score" is high, be more aggressive from the start.
            if pixel_score > pixel_budget and width > 320: # only if score is high and res is not already low
                await retry_send_text_by_msg(msg, "Video has high resolution/fps. Applying smart compression...")

                # Reduce width and fps based on how much it's over budget
                reduction_factor = (pixel_score / pixel_budget) ** 0.5  # sqrt makes reduction less drastic

                scale_width = int(width / reduction_factor)
                final_fps = int(fps / reduction_factor)

                # Clamp to reasonable values
                scale_width = max(min(scale_width, 480), 320)  # Clamp between 320p and 480p width
                final_fps = max(min(final_fps, 25), 15)  # Clamp between 15 and 25 fps
            else:
                # Looks good, use default high-quality settings
                scale_width = min(width, 480)  # Don't upscale, but cap at 480p for sanity
                final_fps = min(fps, 25)  # 25 fps is a good default for video

            await retry_send_text_by_msg(msg, f'Converting to silent mp4 ({scale_width}p width, {final_fps} fps)...')

            _run_ffmpeg_convert_mp4(infile_path, out_mp4, final_fps, scale_width)

            initial_size = out_mp4.stat().st_size
            logger.info('MP4 generated: %s (size=%.1fKB)', out_mp4.name, initial_size / 1024)

            # Check against Telegram's hard limit
            if initial_size > MAX_FILE_SIZE:
                await msg.reply_text(f'Generated video is too large to send ({initial_size / 1024 / 1024:.1f} MB). Try a shorter clip.')
                return

            await retry_send_text_by_msg(msg, f'Sending animation: {out_mp4.name} ({initial_size/1024:.1f} KB).')

            # Send the resulting animation
            await retry_send_animation(
                context.bot,
                msg.chat_id,
                str(out_mp4),
                duration=int(duration),
                width=int(width),
                height=int(height)
            )
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
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, handle_video))

    # Register a global error handler so uncaught exceptions are logged and optionally reported to users
    app.add_error_handler(global_error_handler)

    logger.info('Starting bot...')
    app.run_polling()


if __name__ == '__main__':
    main()