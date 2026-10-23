import inspect, re
import trl, peft, torch
from trl import DPOConfig, DPOTrainer

print("trl", trl.__version__, "| peft", peft.__version__, "| torch", torch.__version__)
params = inspect.signature(DPOConfig.__init__).parameters
want = ["output_dir", "max_steps", "logging_steps", "save_strategy", "report_to",
        "lr_scheduler_type", "warmup_ratio", "warmup_steps", "seed", "disable_dropout",
        "model_init_kwargs", "loss_type", "beta", "eval_strategy", "save_total_limit"]
for k in want:
    p = params.get(k)
    print("DPOConfig." + k + ":", "不存在" if p is None else "預設=" + str(p.default))

src_file = inspect.getsourcefile(DPOTrainer)
print("原始碼:", src_file)
pat = re.compile("is_conversational|apply_chat_template|disable_adapter|get_peft_model|ref_model is None")
for i, line in enumerate(open(src_file, encoding="utf-8"), 1):
    if pat.search(line):
        print(i, line.rstrip()[:120])
print("END")
