import os
import logging
import tempfile
import shutil
import subprocess
from pathlib import Path
import asyncio


from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.error import TimedOut, NetworkError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Environment
TOKEN = os.getenv('VIDEO2GIF_BOT_TOKEN')

if not TOKEN:
	logger.error('Environment variable VIDEO2GIF_BOT_TOKEN is not set')
	raise SystemExit('VIDEO2GIF_BOT_TOKEN required')

# Maximum file size for Telegram (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024  


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


async def compress_gif(input_path, max_size=MAX_FILE_SIZE):
    """Compress GIF if it's too large"""
    if os.path.getsize(input_path) <= max_size:
        return input_path
        
    output_path = input_path.replace('.gif', '_compressed.gif')
    try:
        # Reduce colors and quality until file size is acceptable
        cmd = [
            'convert', input_path,
            '-colors', '128',
            '-quality', '60',
            output_path
        ]
        subprocess.run(cmd, check=True)
        
        if os.path.getsize(output_path) <= max_size:
            return output_path
            
        # If still too large, reduce resolution
        cmd = [
            'convert', input_path,
            '-resize', '50%',
            '-colors', '64',
            '-quality', '50',
            output_path
        ]
        subprocess.run(cmd, check=True)
        return output_path
    except subprocess.CalledProcessError:
        return input_path


async def retry_send_animation(bot, chat_id, animation_path, max_retries=3):
    """Retry sending animation with exponential backoff"""
    for attempt in range(max_retries):
        try:
            with open(animation_path, 'rb') as animation:
                await bot.send_animation(
                    chat_id=chat_id,
                    animation=animation,
                    read_timeout=60,
                    write_timeout=60
                )
            return True
        except (TimedOut, NetworkError) as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(2 ** attempt)
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

		# Download file
		f = await file_obj
		infile_path = in_path.with_suffix(Path(f.file_path).suffix or '.mp4')
		await f.download_to_drive(custom_path=str(infile_path))

		# Conversion parameters (reasonable defaults)
		max_duration = 15  # seconds, trim longer videos
		fps = 15
		scale_width = 480  # scale width, keep aspect ratio

		try:
			await msg.reply_text('Converting to GIF (this may take a few seconds)...')
			# Generate palette and then use it for better colors
			_run_ffmpeg_generate_palette(infile_path, palette, max_duration, fps, scale_width)
			_run_ffmpeg_palette_use(infile_path, palette, out_gif, max_duration, fps, scale_width)

			# Check size (Telegram bots have file size limits — commonly 50 MB)
			max_send_size = 50 * 1024 * 1024
			size = out_gif.stat().st_size
			if size > max_send_size:
				await msg.reply_text(f'GIF is too large to send ({size/1024/1024:.1f} MB). Try a shorter clip.')
				return

			# Compress if needed
			final_gif = await compress_gif(out_gif)

			# Send the resulting GIF
			await retry_send_animation(context.bot, msg.chat_id, final_gif)
			await msg.reply_text('Done!')
		except subprocess.CalledProcessError as e:
			logger.exception('ffmpeg failed: %s', e)
			await msg.reply_text('Conversion failed. ffmpeg error.')
		except Exception as e:
			logger.exception('Unexpected error: %s', e)
			await msg.reply_text('An unexpected error occurred during conversion.')


def main():
	app = ApplicationBuilder().token(TOKEN).build()

	app.add_handler(CommandHandler('start', start))
	app.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO | filters.ANIMATION, handle_video))

	logger.info('Starting bot...')
	app.run_polling()


if __name__ == '__main__':
	main()

