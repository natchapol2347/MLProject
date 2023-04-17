import load_model as lm
model = lm.load_model()
image = lm.processed_image_from_pic(r"C:\Users\USER\MLProject\words\l04\l04-174\l04-174-01-03.png")
print(image)
pred = lm.pred(model,image)
print(pred)