from transformers import MarianMTModel, MarianTokenizer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "Helsinki-NLP/opus-mt-vi-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)


def translate(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(
        device
    )
    translated = model.generate(**inputs, early_stopping=True)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)


if __name__ == "__main__":
    vi_texts = [
        "Cô cho biết: trước giờ tôi không đến phòng tập công cộng, mà tập cùng giáo viên Yoga riêng hoặc tự tập ở nhà.",
        "Khi tập thể dục trong không gian riêng tư, tôi thoải mái dễ chịu hơn.",
        "cô cho biết trước giờ tôi không đến phòng tập công cộng mà tập cùng giáo viên yoga riêng hoặc tự tập ở nhà khi tập thể dục trong không gian riêng tư tôi thoải mái dễ chịu hơn",
    ]
    result = translate(vi_texts)
    print(result)
