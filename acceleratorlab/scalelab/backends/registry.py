from scalelab.backends.openai_compat import OpenAICompatAdapter
from scalelab.backends.sglang import SGLangAdapter
from scalelab.backends.tensorrt_llm import TensorRTLLMAdapter
from scalelab.backends.tgi import TGIAdapter
from scalelab.backends.vllm import VLLMAdapter

BACKENDS = {
    "vllm": VLLMAdapter(),
    "sglang": SGLangAdapter(),
    "tgi": TGIAdapter(),
    "openai-compat": OpenAICompatAdapter(),
    "tensorrt-llm": TensorRTLLMAdapter(),
}
