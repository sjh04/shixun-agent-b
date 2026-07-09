# git detached HEAD 恢复

- memory_id: `mem_conv_c15`
- conversation_id: `eval_c15`
- created_or_updated_at: `2026-06-07T10:00:00+00:00`

## Final Answer

git 提示 detached HEAD 是因为直接 checkout 了某个提交号。想保留当前修改就用 git switch -c 新分支名 把工作现场变成新分支；不需要保留就 git switch main直接回主分支。提交在游离状态下做的 commit 不会丢，reflog 里能找回。
