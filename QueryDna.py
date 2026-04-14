import timm
model = timm.create_model('resnet50', pretrained=True)
# This is the modern way to see the model's "birth certificate"
print(model.pretrained_cfg['tag'])