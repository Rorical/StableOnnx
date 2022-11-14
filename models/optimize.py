import onnx
import onnxoptimizer

src_onnx = 'decoder_fp16.onnx'
opt_onnx = 'decoder_fp16_opt.onnx'

model = onnx.load(src_onnx)

model = onnxoptimizer.optimize(model, ['eliminate_nop_cast', 'eliminate_nop_concat', 'eliminate_if_with_const_cond', 'eliminate_identity', 'eliminate_duplicate_initializer', 'eliminate_deadend', 'eliminate_nop_dropout', 'eliminate_nop_expand', 'eliminate_nop_flatten', 'eliminate_nop_monotone_argmax', 'eliminate_nop_pad', 'eliminate_nop_reshape', 'eliminate_nop_split', 'eliminate_nop_transpose', 'eliminate_shape_gather', 'eliminate_shape_op', 'eliminate_slice_after_shape', 'eliminate_unused_initializer', 'fuse_add_bias_into_conv', 'fuse_bn_into_conv', 'fuse_concat_into_reshape', 'fuse_consecutive_concats', 'fuse_consecutive_log_softmax', 'fuse_consecutive_reduce_unsqueeze', 'fuse_consecutive_squeezes', 'fuse_consecutive_transposes', 'fuse_matmul_add_bias_into_gemm', 'fuse_pad_into_pool', 'fuse_transpose_into_gemm', 'replace_einsum_with_matmul', 'nop'])

onnx.save(model, opt_onnx)