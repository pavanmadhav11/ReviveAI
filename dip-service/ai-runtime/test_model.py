import cv2
import subprocess
from pathlib import Path

runtime = Path(__file__).parent / 'runtime'
image = cv2.imread(r'C:\Users\srini\AppData\Local\Temp\codex-clipboard-4b1d79e3-6737-44c5-9fcb-f3575eccecb5.png')
if image is None:
    raise SystemExit('Could not load the attached comparison image')
crop = image[116:290, 21:325]
source = runtime / 'reviveai-input.png'
target = runtime / 'reviveai-ai-output.png'
cv2.imwrite(str(source), crop)
run = subprocess.run([str(runtime / 'realesrgan-ncnn-vulkan.exe'), '-i', str(source), '-o', str(target), '-n', 'realesrgan-x4plus', '-s', '2', '-t', '128', '-m', str(runtime / 'models')], cwd=runtime, capture_output=True, text=True, timeout=120)
print('exit', run.returncode, run.stdout, run.stderr)
result = cv2.imread(str(target))
print('result', None if result is None else result.shape)
if run.returncode or result is None:
    raise SystemExit(1)
