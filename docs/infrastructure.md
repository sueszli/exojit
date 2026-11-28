# Compute

You log into a login node and stay there. `claude` runs there too, and anything that computes is sent to slurm:

```
                                ┌──sbatch──► compute node (cpu)
                                │                   │
                                │              tests, uv sync
laptop ──ssh──► login node ─────┤
                     │          └──sbatch──► compute node (gpu)
              tmux, claude                          │
                                              ml experiments
```

- the login node is capped at 1 core and 8 GB per user, but it is the only shell with no walltime, so `claude` lives there inside `tmux` and lives until the node reboots
- anything that computes, installs, or loads a model goes through `srun` or `sbatch`, never in the agent's own shell
- `login.rc.fas.harvard.edu` round-robins over unequal nodes. if the shell is slow, `ssh boslogin0N` to another
- cpu work from the login node: `./scripts/allocate-cpu.sh uv run --frozen pytest`. the helper requests 32 GB on `test` and defaults to 1 CPU and one hour. use `--cpus` for parallel commands, `--duration` for walltime, and `srun` directly for a different partition or memory. equivalent `salloc` flags give a shell to iterate in by hand, which ends at walltime, so not for `claude`
- gpu shell, only to debug something that needs a visible gpu: `salloc -p gpu_test --gres=gpu:nvidia_a100_3g.20gb:1 -c 1 --mem=64G -t 0-02:00`
- Workstation execution is unsupported. CPU helpers require Slurm. GPU helpers reuse a visible GPU only inside a Slurm job. LLM helpers always submit their server through Slurm.
- allocation helpers take `./scripts/allocate-<gpu>.sh [--duration d-hh:mm] [--cpus n] <command>`, default to two hours and one core, and reuse an existing Slurm GPU allocation when a GPU is visible.
- otherwise they request one GPU by default and run through `scripts/env.sh`.
- `scripts/allocate-h200.sh` also accepts `--gpus n` to run the command once per node, with one H200 per node.
- `gpu_test` (`allocate-a100-20gb.sh`) and `seas_gpu` (`allocate-a100-80gb.sh`, `allocate-h200.sh`) start fastest. every other helper uses `gpu_requeue`, which can queue for a long time and preempts and requeues jobs, so use it only for restartable work
- gpu batch job without a script, from the project root. for an H200 swap `gpu_test` for `seas_gpu --gres=gpu:nvidia_h200:1`:

  ```bash
  mkdir -p /n/netscratch/$(id -gn)/Lab/$USER/logs && sbatch -J tiny-harness -p gpu_test --gres=gpu:nvidia_a100_3g.20gb:1 -c 1 --mem=64G -t 0-02:00 -o /n/netscratch/$(id -gn)/Lab/$USER/logs/%x-%j.out --wrap './scripts/env.sh timeout 1h nvidia-smi -L'
  ```

- `test` and `gpu_test` cap at 12 h, but availability depends on load and maintenance. Check `sinfo` rather than assuming they start immediately. `gpu_test` is A100 MIG 20 GB only
- size memory from `sacct -j <id> --format=JobID,MaxRSS,ReqMem,Elapsed -P`
- size cores from `seff <id>`. request more than one only for code that runs in parallel, such as worker pools or `pytest -n`. FASRC flags jobs that use one of several allocated cores
- bind servers on a compute node to `127.0.0.1` and reach them with `ssh -N -L <local>:127.0.0.1:<port> <host>` while you hold a job there
- `scripts/run-qwen*.sh` tunnel the server to the login node, so `LLM_URL` only works for commands that run on the login node, not inside an allocation
- a server that must outlive the walltime submits with `--signal=B:TERM@120 --dependency=singleton`, traps TERM, and resubmits itself

# Storage

Every node mounts the same three nfs filesystems, and each node also has its own local `/tmp`:

```
login node       compute node (cpu)     compute node (gpu)
    │                   │                     │
    │                 /tmp                  /tmp          venv, caches. per node, per job
    │                   │                     │
    └───────────────────┼─────────────────────┘
                        │
                    /n/home06/$USER       code, dotfiles
                    /n/netscratch/...     weights, datasets, logs
                    /n/holylabs/...       results, checkpoints
```

- code and dotfiles live in `$HOME`. it is `nfs`, small, snapshotted, and visible from every node
- venvs and scratch files live on `/tmp`. it is node-local and empty at job start, so the venv is rebuilt per job
- download caches (uv, python, pip, triton, inductor, vllm) live on netscratch, so packages are downloaded once
- model weights, datasets, and job logs live on `/n/netscratch/$(id -gn)/Lab/$USER`. it is `nfs`, large, and purged. `HF_HOME` goes here, download from a compute node
- results and checkpoints live on `/n/holylabs/LABS/$(id -gn)/Lab/$USER`. it is `nfs` and backed up
- every tool defaults to `$HOME`. `scripts/env.sh` points them at `/tmp` or netscratch and sets `OMP_NUM_THREADS` to the cores the job has. every helper runs commands through it. in a hand-made `srun`, `sbatch`, or `salloc` shell, prefix commands with `./scripts/env.sh` or `source` it
