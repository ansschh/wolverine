"""Auxiliary "clean" predictors that feed the evidence builder.

Each predictor MUST itself be trained under the same decontamination
protocol as the main system (see spec §3.6 / §8.5). v1 ships scaffolding;
training entry points activate alongside Stage-1 pretrain.
"""
