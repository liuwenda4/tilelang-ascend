import tilelang
import tilelang.language as T
from tvm import tir


def test_shmem_mte_quiet_intrinsic_and_codegen():
    call = T.shmem_mte_quiet()
    assert call.op.same_as(tir.op.Op.get("tl.ascend_shmem_mte_quiet"))
    assert len(call.args) == 0

    @T.prim_func
    def main():
        with T.Kernel(1, is_npu=True):
            with T.Scope("V"):
                T.shmem_mte_quiet()

    source = tilelang.lower(main, target="ascendc").kernel_source
    assert "tl::ascend::shmem_mte_quiet();" in source


def test_shmem_signal_wait_until_intrinsic_and_codegen():
    @T.prim_func
    def main(signal: T.Tensor((8,), "float32")):
        with T.Kernel(1, is_npu=True):
            with T.Scope("V"):
                T.shmem_signal_wait_until(signal, 0, 0, 1)

    source = tilelang.lower(main, target="ascendc").kernel_source
    assert "tl::ascend::shmem_signal_wait_until<float>" in source


def test_shmem_signal_op_intrinsic_and_codegen():
    @T.prim_func
    def main(signal: T.Tensor((8,), "float32")):
        with T.Kernel(1, is_npu=True):
            with T.Scope("V"):
                T.shmem_signal_op(signal, 0, 1, 0, 1)

    source = tilelang.lower(main, target="ascendc").kernel_source
    assert "tl::ascend::shmem_signal_op<float>" in source
