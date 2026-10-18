"""The DataShop toolbox: DataShop's file formats, and the operations on them.

- :mod:`~kcluster.datashop.student_step`: the student-step file, as the
  contract between a dataset driver and KC validation.
- :mod:`~kcluster.datashop.export`: a raw student-step export, reduced to the
  steps a question bank covers.
- :mod:`~kcluster.datashop.kc_model`: KC-model files, as DataShop exports them
  and takes them back.
- :mod:`~kcluster.datashop.learnsphere`: the results of a LearnSphere workflow.

KCluster's own steps (``tag``, ``fit``, the ``*-datashop-kc`` commands) are
built on it, and so can anything else that reads or writes these files: it
imports nothing of KCluster beyond :mod:`kcluster.core` and
:mod:`kcluster.io.jsonl`, so the base install is enough to use it.
"""
