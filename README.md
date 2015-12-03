# Neural machine translation

Repository to collect code for neural machine translation internally at MILA. The short-term objective is to have an attention-based model working on multiple GPUs (see [#6](https://github.com/bartvm/nmt/issues/6)). My proposal is to base the model code of Cho's for now (see [#1](https://github.com/bartvm/nmt/issues/1), because it has simpler internals than Blocks that we can hack away at if needed for multi-GPU.

To have a central collection of research ideas and discussions, please create issues and comment on them.
