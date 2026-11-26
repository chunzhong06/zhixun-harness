"""验证跨轮可选状态只影响计算量；用批量敏感的数值区分末尾不足一批的情况。"""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from harness.learning.inference.bearing_validation_prefix import FILENAME, QUERY_ROWS, predict_with_prefix
from harness.learning.inference.teacher_cache import prefetch_teacher


class BearingValidationPrefixTest(unittest.TestCase):
    def run_query(self, raw, prior, target, *, namespace="teacher-one-thread", initial=None):
        target.mkdir()
        memory = {} if initial is None else dict(initial)
        sizes = []
        def kernel(x):
            sizes.append(len(x))
            return x[:,0] + len(x)
        def current(x):
            prefetch_teacher(memory,x,kernel,batch_size=QUERY_ROWS)
            return np.array([memory[row.tobytes()] for row in x])
        result, reused = predict_with_prefix(raw,namespace,memory,current,prior,target)
        return result,reused,sizes

    def test_appended_rows_and_duplicate_keep_batch_boundaries(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            original=np.arange(135,dtype=np.float64).reshape(-1,1)
            original=np.vstack([original[:3],original])
            self.run_query(original,None,root/"prior")
            current=np.vstack([original,[[135.],[136.]]])
            expected,_,_=self.run_query(current,None,root/"control")
            result,reused,sizes=self.run_query(current,root/"prior",root/"candidate")
            np.testing.assert_array_equal(result,expected)
            self.assertEqual((reused,sizes),(128,[9]))

    def test_other_teacher_reorder_and_damaged_state_recompute(self):
        original=np.arange(132,dtype=np.float64).reshape(-1,1)
        for condition in ("new_teacher","reorder","damaged","truncated"):
            with self.subTest(condition=condition), TemporaryDirectory() as folder:
                root=Path(folder)
                self.run_query(original,None,root/"prior")
                current=original.copy()
                state=root/"prior"/FILENAME
                if condition=="reorder":
                    current[[0,1]]=current[[1,0]]
                if condition=="damaged":
                    values=bytearray(state.read_bytes())
                    values[-1]^=1
                    state.write_bytes(values)
                if condition=="truncated":
                    state.write_bytes(state.read_bytes()[:-1])
                namespace="other" if condition=="new_teacher" else "teacher-one-thread"
                expected,_,_=self.run_query(current,None,root/"control",namespace=namespace)
                result,reused,sizes=self.run_query(current,root/"prior",root/"candidate",namespace=namespace)
                self.assertEqual((reused,sizes),(0,[128,4]))
                np.testing.assert_array_equal(result,expected)
