"""Demo: run the pipeline end-to-end on a sample dialogue.

Offline check:   python demo.py
Real LLM check:  python demo.py anthropic   (needs ANTHROPIC_API_KEY)
                 python demo.py openai      (needs OPENAI_API_KEY)
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from cps_classifier import (
    CPSClassifier, Message, build_user_prompt, extract_contexts,
    load_codebook, make_client,
)

DIALOGUE = [
    Message("Lion", "What's everyone's voltage right now?"),
    Message("Tiger", "I currently have 2.0 volts with a 180 ohm resistor."),
    Message("Bear", "Mine says 1.5 volts."),
    Message("Lion", "Everyone try to make all of your resistors equal to 100 ohms."),
    Message("Tiger", "Ok."),
    Message("Bear", "I got 325, I changed it to 330."),
    Message("Tiger", "It keeps telling me mine's incorrect."),
    Message("Lion", "I think so."),
]

load_dotenv()

provider = sys.argv[1] if len(sys.argv) > 1 else "mock"
codebook = load_codebook(Path(__file__).parent / "codebook_andrews_todd.json")

# Show the assembled prompt for one message so you can inspect the design.
t = 6  # Tiger: "It keeps telling me mine's incorrect."
cog, soc = extract_contexts(DIALOGUE, t, w_cognitive=2, w_social=1)
print("=" * 70)
print(f"Example user prompt for message {t}:")
print("=" * 70)
print(build_user_prompt(cog, soc, DIALOGUE[t]))
print("=" * 70, "\n")

clf = CPSClassifier(make_client(provider), codebook)
for res in clf.classify_dialogue(DIALOGUE):
    m = DIALOGUE[res.index]
    print(f"[{res.index}] {m.speaker:>6}: {m.text!r:55} -> "
          f"{res.label} ({res.dimension})")
