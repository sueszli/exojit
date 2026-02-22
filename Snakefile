import platform
import subprocess

configfile: "config.yaml"


SCALAR_PROCS=[
    "conv1d",
    "gemm",
]
LEVEL_1_PROCS = [
    "asum",
    "axpy",
    "dot",
    "exo_copy",
    "rot",
    # "scal",
    "swap",
]


# ---- EXOCC ----

rule exocc_compile_exocc:
    input:
        "benchmarks/{level}/{proc}.py",
    output:
        "build/exocc/{level}/{proc}.c",
        "build/exocc/{level}/{proc}.h",
    shell:
        """
        exocc -o build/exocc/{wildcards.level} benchmarks/{wildcards.level}/{wildcards.proc}.py
        """

rule exocc_compile_gcc:
    input:
        "build/exocc/{level}/{proc}.c",
        "build/exocc/{level}/{proc}.h",
    output:
        "build/exocc/{level}/{proc}.o",
    shell:
        """
        gcc -O3 -mavx -mfma -mavx2 -c build/exocc/{wildcards.level}/{wildcards.proc}.c -o build/exocc/{wildcards.level}/{wildcards.proc}.o -save-temps=obj
        """

# ---- EXOMLIR ----

rule exomlir_compile_exomlir:
    input:
        "benchmarks/{level}/{proc}.py",
    output:
        "build/exomlir/{level}/{proc}.mlir",
    shell:
        """
        .venv/bin/python -m exomlir.main -o build/exomlir/{wildcards.level}/ benchmarks/{wildcards.level}/{wildcards.proc}.py --target llvm --prefix exomlir
        """

rule exomlir_compile_mliropt:
    input:
        "build/exomlir/{level}/{proc}.mlir",
    output:
        "build/exomlir/{level}/{proc}.lowered.mlir",
    shell:
        """
        mlir-opt -convert-vector-to-llvm --convert-to-llvm -cse -canonicalize build/exomlir/{wildcards.level}/{wildcards.proc}.mlir > build/exomlir/{wildcards.level}/{wildcards.proc}.lowered.mlir
        """

rule exomlir_compile_mlirtranslate:
    input:
        "build/exomlir/{level}/{proc}.lowered.mlir",
    output:
        "build/exomlir/{level}/{proc}.ll",
    shell:
        """
        mlir-translate -mlir-to-llvmir build/exomlir/{wildcards.level}/{wildcards.proc}.lowered.mlir > build/exomlir/{wildcards.level}/{wildcards.proc}.ll
        """

rule exomlir_compile_clang:
    input:
        "build/exomlir/{level}/{proc}.ll",
    output:
        "build/exomlir/{level}/{proc}.o",
    shell:
        """
        clang -O3 -mavx -mfma -mavx2 -Wall -Wextra -pedantic -c build/exomlir/{wildcards.level}/{wildcards.proc}.ll -o build/exomlir/{wildcards.level}/{wildcards.proc}.o --save-temps=obj
        """

# ---- CORRECTNESS ----

rule benchmark_compile_correctness:
    input:
        "build/exocc/{level}/{proc}.o",
        "build/exomlir/{level}/{proc}.o",
        "benchmarks/{level}/{proc}.correctness.cpp",
    output:
        "build/correctness/{level}/{proc}.x",
    shell:
        """
        clang++ -O3 -mavx -mfma -mavx2 -fuse-ld=lld -fsanitize=address,undefined \
            -Ibuild \
            -o build/correctness/{wildcards.level}/{wildcards.proc}.x \
            build/exocc/{wildcards.level}/{wildcards.proc}.o \
            build/exomlir/{wildcards.level}/{wildcards.proc}.o \
            benchmarks/{wildcards.level}/{wildcards.proc}.correctness.cpp
        """

rule benchmark_run_correctness:
    input:
        "build/correctness/{level}/{proc}.x",
    output:
        "build/correctness/{level}/{proc}.out",
    shell:
        """
        ./build/correctness/{wildcards.level}/{wildcards.proc}.x > build/correctness/{wildcards.level}/{wildcards.proc}.out
        """

# ---- BENCHMARKS ----

rule benchmark_compile_harnesses:
    input:
        "build/exocc/{level}/{proc}.o",
        "build/exomlir/{level}/{proc}.o",
        "benchmarks/{level}/{proc}.harness.cpp",
        "build/correctness/{level}/{proc}.out"
    output:
        "build/harnesses/{level}/{proc}.x",
    shell:
        """
        clang++ -O3 -mavx -mfma -mavx2 -fuse-ld=lld -fsanitize=address,undefined \
            -Ibuild \
            $(pkg-config --cflags benchmark) \
            $(pkg-config --libs benchmark) -lpthread -lm -lstdc++ \
            -o build/harnesses/{wildcards.level}/{wildcards.proc}.x \
            build/exocc/{wildcards.level}/{wildcards.proc}.o \
            build/exomlir/{wildcards.level}/{wildcards.proc}.o \
            benchmarks/{wildcards.level}/{wildcards.proc}.harness.cpp
        """

rule benchmark_run_harnesses:
    input:
        "build/harnesses/{level}/{proc}.x",
    output:
        "build/results/{level}/{proc}.csv",
    shell:
        """
        ./build/harnesses/{wildcards.level}/{wildcards.proc}.x \
            --benchmark_format=csv \
            --benchmark_report_aggregates_only=false \
            --benchmark_repetitions=16 \
            > build/results/{wildcards.level}/{wildcards.proc}.csv
        """

rule benchmark_process_results:
    input:
        "build/results/{level}/{proc}.csv",
    output:
        "results/data/{level}/{proc}.processed.csv",
    shell:
        """
        python3 tools/process-results.py {input} {output}
        """

rule benchmark_plot_results:
    input:
        "results/data/{level}/{proc}.processed.csv",
    output:
        "results/plots/{level}/{proc}.png",
    shell:
        """
        python3 tools/plot-benchmark-results.py {input} {wildcards.level} {wildcards.proc} {output}
        """

rule heatmaps:
    input:
        expand(
            "results/data/scalar/{proc}.processed.csv",
            proc=SCALAR_PROCS
        ),
        expand(
            "results/data/level1/{proc}.processed.csv",
            proc=LEVEL_1_PROCS
        ),
        expand(
            "results/data/level1-unopt/{proc}.processed.csv",
            proc=LEVEL_1_PROCS
        ),
    output:
        "results/plots/scalar/heatmap.png",
        "results/plots/level1/heatmap.png",
        "results/plots/level1-unopt/heatmap.png",
    shell:
        """
        python3 tools/plot-heatmaps.py results/data/scalar/ results/plots/scalar/heatmap.png && \
        python3 tools/plot-heatmaps.py results/data/level1/ results/plots/level1/heatmap.png && \
        python3 tools/plot-heatmaps.py results/data/level1-unopt/ results/plots/level1-unopt/heatmap.png
        """

rule all:
    input:
        # plots
        expand(
            "results/plots/scalar/{proc}.png",
            proc=SCALAR_PROCS
        ),
        expand(
            "results/plots/level1/{proc}.png",
            proc=LEVEL_1_PROCS
        ),
        expand(
            "results/plots/level1-unopt/{proc}.png",
            proc=LEVEL_1_PROCS
        ),
        # heatmaps
        "results/plots/scalar/heatmap.png",
        "results/plots/level1/heatmap.png",
        "results/plots/level1-unopt/heatmap.png",

# rule benchmark_count_instrs:
#     input:
#         expand(
#             "build/{compiler}/{level}/{proc}.bc",
#             compiler=["exocc", "exomlir"],
#             level=config["levels"],
#             proc=config["procs"]
#         )
#     output:
#         "build/instrcount.csv"
#     params:
#         opt=config["opt"]
#     shell:
#         """
#         python3 tools/count-instructions.py {params.opt} ./build > build/instrcount.csv
#         """

# rule benchmark_plot_instr_counts:
#     input:
#         "build/instrcount.csv"
#     output:
#         "build/plots/level1/instcount.png",
#         "build/plots/level2/instcount.png",
#     shell:
#         """
#         python3 tools/plot-instruction-counts.py
#         """

# rule all:
#     input:
#         # inst counts
#         "build/plots/level1/instcount.png",
#         "build/plots/level2/instcount.png",
#         # heatmaps
#         "build/plots/level1/heatmap.png",
#         "build/plots/level2/heatmap.png",
#         # correctness
#         expand(
#             "build/correctness/{level}/{proc}.out",
#             level=config["levels"],
#             proc=config["procs"]
#         ),
#         # benchmarks
#         expand(
#             "build/plots/{level}/{proc}.png",
#             level=config["levels"],
#             proc=config["procs"]
#         ),
