import tensorflow as tf
from Lib.Model import JDNDMSR_1
from Config import Cfg
import os
from keras_flops import get_flops
import kerop

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


model = JDNDMSR_1.get_model(
    initializer=Cfg.initializers,
    filters=Cfg.model_filters,
    depth=Cfg.model_depth,
)

model.summary()
flops = get_flops(model, batch_size=1)
print(f"FLOPS: {flops / 10 ** 9:.03} G")

# # run profile
# log = True
# layer_name, layer_flops, inshape, weights = kerop.profile(model, log)

# # visualize results
# for name, flop, shape, weight in zip(layer_name, layer_flops, inshape, weights):
#     print("layer:", name, shape, " MegaFLOPS:",
#           flop/1e6, " MegaWeights:", weight/1e6)

# print("Total FLOPS[GFLOPS]:", sum(layer_flops)/1e9)
