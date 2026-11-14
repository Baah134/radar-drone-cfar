# `make all` reruns everything (about 1-2 hours; needs Vivado for RTL simulation and the
# Vivado cross-check). `make figures` regenerates RESULTS.md from committed results.
# Windows without make: python reproduce.py <target>.
PY ?= python
RUN = $(PY) -W ignore -m

.PHONY: all test params theory scenes agree rtl-test mutation waveform synth vivado classify figures

all: test params theory scenes agree rtl-test mutation waveform synth vivado classify figures

test:
	$(PY) -m pytest tests -q
params:
	$(RUN) scripts.radar_params
theory:
	$(RUN) scripts.cfar_theory
scenes:
	$(RUN) scripts.scenes --frames 100
agree:
	$(RUN) scripts.hw_agreement --frames 40
rtl-test:
	$(RUN) tb.run_xsim
mutation:
	$(RUN) tb.mutation_test
waveform:
	$(RUN) tb.waveform
synth:
	$(RUN) synth.run_yosys
vivado:
	$(RUN) synth.run_vivado
classify:
	$(RUN) scripts.classify --n 400
figures:
	$(RUN) scripts.write_results
board:
	$(RUN) board.board_test gen
	$(RUN) board.board_test build
	$(RUN) board.board_test program
	$(RUN) board.board_test run --port COM10
