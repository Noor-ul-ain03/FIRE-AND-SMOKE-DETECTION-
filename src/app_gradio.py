"""Module 9a: browser interface. Upload a photo, see fire / smoke probabilities and the decision.
Usage: pip install gradio      then      python src/app_gradio.py
Opens at http://127.0.0.1:7860 (the address is also printed in the terminal).
"""
import gradio as gr

from predict import predict

DISCLAIMER = ("Decision-support only. This model never saw photos without fire or smoke, "
              "so it can raise false alarms. It must not replace certified fire alarms or a person.")


def run(img):
    if img is None:
        return {}, "Upload a photo first."
    r = predict(img)
    probabilities = {"fire": r["p_fire"], "smoke": r["p_smoke"]}
    text = f"### Decision: {r['decision']}\n\n"
    text += (f"Thresholds used: fire {r['thr_fire']:.2f}, smoke {r['thr_smoke']:.2f} "
             f"(tuned on validation data).\n\n")
    if r["needs_human_check"]:
        text += "**Needs human check:** at least one probability is close to its threshold.\n\n"
    text += f"*{DISCLAIMER}*"
    return probabilities, text


demo = gr.Interface(
    fn=run,
    inputs=gr.Image(type="pil", label="Photo"),
    outputs=[gr.Label(num_top_classes=2, label="Probability"), gr.Markdown()],
    title="Fire and Smoke Detector",
    description="Upload a photo. The ensemble of ResNet18 and EfficientNet-B0 estimates whether fire and smoke are visible.",
)

if __name__ == "__main__":
    demo.launch(share=True)
  
