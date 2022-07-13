import tensorflow as tf
import numpy as np
from sklearn import datasets
from random import shuffle

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, Activation
from tensorflow.keras.models import Model

import yaml
import sys

sys.path.insert(0, '/data')
from modelzoo.common.tf.estimator.run_config import CSRunConfig
from modelzoo.common.tf.estimator.cs_estimator import CerebrasEstimator
from modelzoo.common.tf.estimator.cs_estimator_spec import CSEstimatorSpec

from tensorflow.keras.mixed_precision.experimental import Policy

from modelzoo.common.tf.run_utils import (
    check_env,
    get_csrunconfig_dict,
    is_cs,
    save_params,
    update_params_from_args,
)

from modelzoo.fc_mnist.tf.utils import (
    DEFAULT_YAML_PATH,
    get_custom_stack_params,
    get_params,
)

from modelzoo.fc_mnist.tf.run import create_arg_parser, validate_params

try:
    from cerebras.pb.stack.autogen_pb2 import AP_ENABLED
    from cerebras.pb.stack.full_pb2 import FullConfig
except ImportError:
    print('No se import Full config')


iris_dtype = np.dtype([('img', "float32", 4),
                      ('lbl', 'int32', '')])

ir = datasets.load_iris()

X,y = datasets.load_iris(return_X_y=True)
xy = list(zip(X,y))
shuffle(xy)
X,y = zip(*xy)


ir_data_valtn = np.array(X[:50])
ir_data_train = np.array(X[50:])

ir_labl_valtn = np.array(y[:50])
ir_labl_train = np.array(y[50:])
#config.matching.autogen_policy = AP_ENABLED
def input_fn(params, mode=tf.estimator.ModeKeys.TRAIN):
    ir = datasets.load_iris()
    #dtype = Policy('mixed_float16', loss_scale=None)
    ds = tf.data.Dataset.from_tensor_slices(
        (ir.data.astype('float32'), ir.target.astype('int32')))
        #datasets.load_iris(return_X_y=True))
        
    ds = ds.shuffle(1000).repeat().batch(100, drop_remainder=True)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

def build_model(features, labels, mode, params):
    print(f'params = {params}')
    dtype = Policy('mixed_float16', loss_scale=None)
    tf.keras.mixed_precision.experimental.set_policy(dtype)
    tf.keras.backend.set_floatx('float16')
    i = features
    x = Dense(64, activation='relu',dtype=dtype)(i)
    x = Dense(3, activation='softmax',dtype=dtype)(x) # LOGITS
    
    #loss_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=x)
    #loss_per_sample = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=x)
    #loss = tf.reduce_mean(input_tensor=loss_per_sample)
   # loss = tf.keras.losses.SparseCategoricalCrossentropy()
   # tf.compat.v1.summary.scalar('loss', loss)
    lo = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = lo(labels, x)
    loss = tf.reduce_mean(input_tensor=x)
    return loss, x

def build_model2(features, labels, mode, params):
    """
    build_model function that is more similar to the fc_mnist example
    """
    dtype = Policy('mixed_float16', loss_scale=None)
    tf.keras.mixed_precision.experimental.set_policy(dtype)
    tf.keras.backend.set_floatx('float16')
    
    x = features
    x = Dense(64, dtype=dtype)(x)
    x = Activation('relu', dtype=dtype)(x)
    logits = Dense(3, dtype=dtype)(x)
    losses_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, )
    loss = tf.reduce_mean(input_tensor=losses_per_sample)
    tf.compat.v1.summary.scalar('loss', loss)
    return loss, logits

def model_fn(features, labels, mode, params):
    loss, logits = build_model(features, labels, 'TRAIN', {'p': 2})

    #def model_fn(features, labels, mode, params):
    opt = tf.compat.v1.train.AdamOptimizer(
           learning_rate=0.001,
           beta1=0.9,
           beta2=0.999,
           epsilon=1.0e-8,)
    train_op = opt.minimize(loss=loss,
            global_step=tf.compat.v1.train.get_or_create_global_step())
    mode = tf.estimator.ModeKeys.TRAIN
    espec = CSEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            host_call=None)
    return espec

if __name__ == "__main__":
    #param_repo_path = '/home/luis/Paquetes/modelzoo/fc_mnist/tf/configs/params.yaml'
    param_repo_path = '/data/modelzoo/fc_mnist/tf/configs/params.yaml'
    with open(param_repo_path, 'r') as fobj:
        params = yaml.safe_load(fobj)

    parser = create_arg_parser('/home/luis/rm_me_estimator')
    args = parser.parse_args(sys.argv[1:])
    
    runconfig_params = params["runconfig"]
    update_params_from_args(args, runconfig_params)
    validate_params(params)
    # save params for reproducibility
    save_params(params, model_dir=runconfig_params["model_dir"])

    # get runtime configurations
    use_cs = is_cs(runconfig_params)
    csrunconfig_dict = get_csrunconfig_dict(runconfig_params)
    stack_params = get_custom_stack_params(params)
    #stack_params = FullConfig()
    #stack_params.matching.autogen_policy = AP_ENABLED
    
    

    # prep cs1 run environment, run config and estimator
    check_env(runconfig_params)
    est_config = CSRunConfig(
        cs_ip=runconfig_params["cs_ip"],
        stack_params=stack_params,
        **csrunconfig_dict,
    )
    est = CerebrasEstimator(
        model_fn=model_fn,
        model_dir=runconfig_params["model_dir"],
        config=est_config,
        params=params,
    )

    est.compile(input_fn, validate_only=False)
