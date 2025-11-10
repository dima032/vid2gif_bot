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
# Preferred maximum GIF size for conversations (can be tuned via env var, MB)
MAX_GIF_SIZE = int(os.getenv('VIDEO2GIF_MAX_GIF_SIZE_MB', '3')) * 1024 * 1024


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
		return str(input_path)

	if size <= max_size:
		return str(input_path)

	workdir = inp.parent
	gifsicle = shutil.which('gifsicle')
	convert = shutil.which('convert')

	# 1) gifsicle color reduction / optimization
	if gifsicle:
		try:
			last = inp
			for colors in (128, 64, 32, 16, 8):
				out = workdir / f'{inp.stem}_gfs_{colors}.gif'
				cmd = [gifsicle, '--optimize=3', f'--colors={colors}', str(last), '-o', str(out)]
				subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
				if out.exists() and out.stat().st_size <= max_size:
					return str(out)
				last = out
		except subprocess.CalledProcessError:
			pass

	# 2) ImageMagick convert (resize + colors + optimize)
	if convert:
		try:
			for scale_pct in (75, 50, 40):
				for colors in (64, 32, 16):
					out = workdir / f'{inp.stem}_magick_{scale_pct}_{colors}.gif'
					cmd = [
						convert, str(inp), '-coalesce',
						'-resize', f'{scale_pct}%',
						'-colors', str(colors),
						'-layers', 'Optimize',
						str(out)
					]
					subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
					if out.exists() and out.stat().st_size <= max_size:
						return str(out)
		except subprocess.CalledProcessError:
			pass

	# 3) Re-encode with ffmpeg lower fps/width using palette method
	try:
		for fps in (12, 10, 8, 6, 5):
			for width in (480, 360, 240, 160):
				pal = workdir / f'{inp.stem}_pal_{fps}_{width}.png'
				candidate = workdir / f'{inp.stem}_ff_{fps}_{width}.gif'
				try:
					_run_ffmpeg_generate_palette(inp, pal, None, fps, width)
					_run_ffmpeg_palette_use(inp, pal, candidate, None, fps, width)
				except Exception:
					continue
				if candidate.exists() and candidate.stat().st_size <= max_size:
					return str(candidate)
	except Exception:
		pass

	# Nothing worked; return original GIF
	return str(input_path)


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

