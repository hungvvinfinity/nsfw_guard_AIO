# ComfyUI workflow examples

## Load trực tiếp vào giao diện ComfyUI

Các file `*_workflow.json` là workflow giao diện, có thể kéo thả vào canvas hoặc mở bằng **Workflow → Open**:

| File | Mục đích |
| --- | --- |
| [prompt_only_workflow.json](prompt_only_workflow.json) | Test prompt; chỉ cần model text, không cần ảnh hoặc checkpoint tạo ảnh |
| [image_only_workflow.json](image_only_workflow.json) | `LoadImage → NSFW Guard → PreviewImage / SaveImage` |
| [combined_workflow.json](combined_workflow.json) | Kiểm tra prompt trước, sau đó kiểm tra ảnh và preview/lưu khi cả hai qua kiểm duyệt |

1. Cài custom node và dependencies theo [README chính](../README.md#install), sau đó restart ComfyUI.
2. Kéo một file workflow ở trên vào ComfyUI. Bắt đầu với `prompt_only_workflow.json` để test nhanh.
3. Với workflow ảnh, bấm **choose file to upload** trong `LoadImage` để chọn ảnh. `example.png` chỉ là tên mẫu; bạn cần chọn ảnh thực tế trước khi chạy.
4. Bấm **Queue Prompt / Run**. Lần đầu chạy sẽ tải model. Workflow ảnh dùng model công khai `Falconsai/nsfw_image_detection`, không cần xác thực Hugging Face.
5. Khi qua kiểm duyệt, guard trả `{"status": "safe"}` trong history; workflow ảnh sẽ preview và lưu ảnh vào thư mục output với prefix `nsfw_guard_checked`. Kết quả phân loại phụ thuộc model, kể cả với prompt mẫu.

Để thử nhánh chặn, đặt `prompt_sensitivity = 1.0` cho workflow prompt, hoặc `image_sensitivity = 1.0` cho workflow chỉ ảnh, rồi chạy lại. Mọi nội dung được cung cấp sẽ bị chặn ở mức này **sau khi model tải và inference thành công**; ComfyUI hiển thị lỗi `NSFWDetectedError` kèm JSON. Nhánh preview/lưu phía sau guard sẽ không chạy. Đặt lại `0.5` sau khi thử. Với workflow kết hợp, prompt bị chặn sẽ dừng trước bước kiểm tra ảnh.

Workflow chỉ ảnh để trống `prompt`, nên không tải model text. Bạn có thể sửa hai ô model trên guard thành model ID hoặc đường dẫn model local tương thích.

## API workflow examples

All examples use the built-in model defaults without separate model loader nodes or model fields. Duplicate the guard node freely: instances in the same ComfyUI process share cached model and ModelPatcher objects.

These files are the contents of the `prompt` field accepted by ComfyUI's `/prompt` API:

- `prompt_only.json`: checks a prompt without supplying or loading an image classifier.
- `image_only.json`: loads an image and saves it only after moderation. No text classifier is supplied.
- `combined.json`: checks a prompt first, then the image, before saving the image.

Put an image named `example.png` in ComfyUI's input directory, or change the `LoadImage` filename. The image examples use the public `Falconsai/nsfw_image_detection` model and do not require Hugging Face authentication. Start ComfyUI with this custom node installed, then submit an example from the repository root:

```python
import json
from pathlib import Path
from urllib.request import Request, urlopen

workflow = json.loads(Path("examples/prompt_only.json").read_text())
request = Request(
    "http://127.0.0.1:8188/prompt",
    data=json.dumps({"prompt": workflow}).encode(),
    headers={"Content-Type": "application/json"},
)
with urlopen(request) as response:
    queued = json.load(response)
print(queued)
```

Inspect `/history/<prompt_id>` after execution. Safe results appear under the guard node's `text` output. A blocked workflow contains an `execution_error` status message whose `exception_message` is the detection JSON. Set the relevant sensitivity to `1.0` to demonstrate termination using any input; then restore the desired sensitivity.
