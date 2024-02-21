import torch
import clip

from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

############################### starry night #########################################
original_image = preprocess(Image.open("./MUNIT_CLIP/tubingen.jpg")).unsqueeze(0).to(device)
style_image = preprocess(Image.open("./MUNIT_CLIP/sty2.jpg")).unsqueeze(0).to(device)
transfered_image = preprocess(Image.open("./MUNIT_CLIP/100tubingen_huggingface.png")).unsqueeze(0).to(device)
# not_same_transfer1 = preprocess(Image.open("./MUNIT_CLIP/tubingen_shipwreck.png")).unsqueeze(0).to(device)
# not_same_transfer2 = preprocess(Image.open("./MUNIT_CLIP/tubingen_scream.png")).unsqueeze(0).to(device)


############################### La femme au chapeau #########################################
# original_image = preprocess(Image.open("./MUNIT_CLIP/diff_stage_transfer/1.jpg")).unsqueeze(0).to(device)
# style_image = preprocess(Image.open("./MUNIT_CLIP/diff_stage_transfer/sty0.jpg")).unsqueeze(0).to(device)
# transfered_image = preprocess(Image.open("./MUNIT_CLIP/diff_stage_transfer/out.png")).unsqueeze(0).to(device)
# ti1 = preprocess(Image.open("./MUNIT_CLIP/diff_stage_transfer/out_100.png")).unsqueeze(0).to(device)
# ti2 = preprocess(Image.open("./MUNIT_CLIP/diff_stage_transfer/out_200.png")).unsqueeze(0).to(device)
# ti3 = preprocess(Image.open("./MUNIT_CLIP/diff_stage_transfer/out_300.png")).unsqueeze(0).to(device)
# ti4 = preprocess(Image.open("./MUNIT_CLIP/diff_stage_transfer/out_400.png")).unsqueeze(0).to(device)
# ti5 = preprocess(Image.open("./MUNIT_CLIP/diff_stage_transfer/out_500.png")).unsqueeze(0).to(device)
# ti6 = preprocess(Image.open("./MUNIT_CLIP/diff_stage_transfer/out_600.png")).unsqueeze(0).to(device)
# ti7 = preprocess(Image.open("./MUNIT_CLIP/diff_stage_transfer/out_700.png")).unsqueeze(0).to(device)
# ti8 = preprocess(Image.open("./MUNIT_CLIP/diff_stage_transfer/out_800.png")).unsqueeze(0).to(device)
# ti9 = preprocess(Image.open("./MUNIT_CLIP/diff_stage_transfer/out_900.png")).unsqueeze(0).to(device)

############################### golden gate #########################################
# original_image = preprocess(Image.open("./MUNIT_CLIP/goldengate/golden_gate.jpg")).unsqueeze(0).to(device)
# style_image = preprocess(Image.open("./MUNIT_CLIP/goldengate/frida_kahlo.jpg")).unsqueeze(0).to(device)
# transfered_image = preprocess(Image.open("./MUNIT_CLIP/goldengate/goldengate_huggingface.png")).unsqueeze(0).to(device)

############################### brad pitt #########################################
# original_image = preprocess(Image.open("./MUNIT_CLIP/brad_pitt/brad_pitt.jpg")).unsqueeze(0).to(device)
# style_image = preprocess(Image.open("./MUNIT_CLIP/brad_pitt/picasso_selfport1907.jpg")).unsqueeze(0).to(device)
# ti1 = preprocess(Image.open("./MUNIT_CLIP/brad_pitt/pitt_picasso_content_5_style_10.png")).unsqueeze(0).to(device)
# ti2 = preprocess(Image.open("./MUNIT_CLIP/brad_pitt/pitt_picasso_content_1_style_10.png")).unsqueeze(0).to(device)
# ti3 = preprocess(Image.open("./MUNIT_CLIP/brad_pitt/pitt_picasso_content_01_style_10.png")).unsqueeze(0).to(device)
# ti4 = preprocess(Image.open("./MUNIT_CLIP/brad_pitt/pitt_picasso_content_0025_style_10.png")).unsqueeze(0).to(device)



#preprocess中已经把他们都压缩到224*224

# image = preprocess(Image.open(hyperparameters['data_root'] + '/models')).unsqueeze(0).to(device)
text = clip.tokenize(["Vincent Van Gogh", 'style']).to(device)
# text = clip.tokenize(["starry night the painting", "Vincent Van Gogh", "Pablo Picasso", "Giotto di Bondone", "Leonardo da Vinci", "Paul Cézanne"]).to(device)
#"starry night" Label probs: [[3.9341301e-01 6.0371584e-01 2.0429257e-03 5.9596459e-05 7.1085361e-04  5.7851514e-05]]
#"starry night the painting" Label probs: [[8.46049845e-01 1.53221369e-01 5.18488814e-04 1.51254135e-05  1.80412651e-04 1.46825514e-05]]



with torch.no_grad():
    # if0 = model.encode_image(original_image1)
    if1 = model.encode_image(original_image)
    if2 = model.encode_image(style_image)
    if3 = model.encode_image(transfered_image)
    # if100 = model.encode_image(ti1)
    # if200 = model.encode_image(ti2)
    # if300 = model.encode_image(ti3)
    # if400 = model.encode_image(ti4)
    # if500 = model.encode_image(ti5)
    # if600 = model.encode_image(ti6)
    # if700 = model.encode_image(ti7)
    # if800 = model.encode_image(ti8)
    # if900 = model.encode_image(ti9)
    # if4 = model.encode_image(not_same_transfer1)
    # if5 = model.encode_image(not_same_transfer2)
    tf = model.encode_text(text)


# # 计算 Gram 矩阵
# gram_matrix_t = torch.mm(tf, tf.t())
# # gram_matrix_0 = torch.mm(if0, if0.t())
# gram_matrix_1 = torch.mm(if1, if1.t())
# gram_matrix_2 = torch.mm(if2, if2.t())
# gram_matrix_3 = torch.mm(if3, if3.t())
# gram_matrix_100 = torch.mm(if100, if100.t())
# gram_matrix_200 = torch.mm(if200, if200.t())
# gram_matrix_300 = torch.mm(if300, if300.t())
# gram_matrix_400 = torch.mm(if400, if400.t())
# gram_matrix_500 = torch.mm(if500, if500.t())
# gram_matrix_600 = torch.mm(if600, if600.t())
# gram_matrix_700 = torch.mm(if700, if700.t())
# gram_matrix_800 = torch.mm(if800, if800.t())
# gram_matrix_900 = torch.mm(if900, if900.t())
# gram_matrix_4 = torch.mm(if4, if4.t())
# gram_matrix_5 = torch.mm(if5, if5.t())



# 计算 Gram 矩阵之间的均方误差
# mse_loss0 = torch.nn.functional.mse_loss(gram_matrix_t, gram_matrix_0)#original1
mse_loss1 = torch.nn.functional.mse_loss(gram_matrix_t, gram_matrix_1)#original
mse_loss2 = torch.nn.functional.mse_loss(gram_matrix_t, gram_matrix_2)#style
mse_loss3 = torch.nn.functional.mse_loss(gram_matrix_t, gram_matrix_3)#transfered
# mse_loss100 = torch.nn.functional.mse_loss(gram_matrix_t, gram_matrix_100)#transfered
# mse_loss200 = torch.nn.functional.mse_loss(gram_matrix_t, gram_matrix_200)#transfered
# mse_loss300 = torch.nn.functional.mse_loss(gram_matrix_t, gram_matrix_300)#transfered
# mse_loss400 = torch.nn.functional.mse_loss(gram_matrix_t, gram_matrix_400)#transfered
# mse_loss500 = torch.nn.functional.mse_loss(gram_matrix_t, gram_matrix_500)#transfered
# mse_loss600 = torch.nn.functional.mse_loss(gram_matrix_t, gram_matrix_600)#transfered
# mse_loss700 = torch.nn.functional.mse_loss(gram_matrix_t, gram_matrix_700)#transfered
# mse_loss800 = torch.nn.functional.mse_loss(gram_matrix_t, gram_matrix_800)#transfered
# mse_loss900 = torch.nn.functional.mse_loss(gram_matrix_t, gram_matrix_900)#transfered
# mse_loss4 = torch.nn.functional.mse_loss(gram_matrix_t, gram_matrix_4)#wrong transfer
# mse_loss5 = torch.nn.functional.mse_loss(gram_matrix_t, gram_matrix_5)#wrong transfer


# print(mse_loss0)
print(mse_loss1)
print(mse_loss2)
print(mse_loss3)
# print(mse_loss100)
# print(mse_loss200)
# print(mse_loss300)
# print(mse_loss400)
# print(mse_loss500)
# print(mse_loss600)
# print(mse_loss700)
# print(mse_loss800)
# print(mse_loss900)
# print(mse_loss4)
# print(mse_loss5)



# 将 mse_loss 加入到整体的训练损失中
# total_loss = content_loss + lambda_style * mse_loss  # lambda_style 是风格损失的权重



# def gram_matrix(tensor):
#     _, c, h, w = tensor.size()
#     tensor = tensor.view(c, h * w)
#     gram = torch.mm(tensor, tensor.t())
#     return gram

# # 假设 generated_features 和 reference_features 是生成图像和参考图像的特征图
# generated_grams = [gram_matrix(feature) for feature in generated_features]
# reference_grams = [gram_matrix(feature) for feature in reference_features]

# # 计算风格损失
# style_loss = 0
# for gen_gram, ref_gram in zip(generated_grams, reference_grams):
#     style_loss += torch.nn.functional.mse_loss(gen_gram, ref_gram)

# # 将 style_loss 加入到整体的训练损失中
# total_loss = content_loss + lambda_style * style_loss  # lambda_style 是风格损失的权重
