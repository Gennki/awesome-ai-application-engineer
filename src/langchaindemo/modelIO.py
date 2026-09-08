from src.langchaindemo.model import getModel

model = getModel()
result = model.invoke("你知道？")
print(result.content)
