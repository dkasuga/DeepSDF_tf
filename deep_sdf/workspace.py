#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import json
import os
import tensorflow as tf

model_params_subdir = "ModelParameters"
optimizer_params_subdir = "OptimizerParameters"
latent_codes_subdir = "LatentCodes"
logs_filename = "Logs.ckpt"
reconstructions_subdir = "Reconstructions"
reconstruction_meshes_subdir = "Meshes"
reconstruction_codes_subdir = "Codes"
specifications_filename = "specs.json"
data_source_map_filename = ".datasources.json"
evaluation_subdir = "Evaluation"
sdf_samples_subdir = "SdfSamples"
surface_samples_subdir = "SurfaceSamples"
normalization_param_subdir = "NormalizationParameters"
training_meshes_subdir = "TrainingMeshes"


def load_experiment_specifications(experiment_directory):

  filename = os.path.join(experiment_directory, specifications_filename)

  if not os.path.isfile(filename):
    raise Exception(
        "The experiment directory ({}) does not include specifications file "
        + '"specs.json"'.format(experiment_directory)
    )

  return json.load(open(filename))


def load_model_parameters(experiment_directory, checkpoint, decoder):

  filename = os.path.join(
      experiment_directory, model_params_subdir, checkpoint + ".ckpt"
  )

  if not os.path.isfile(filename):
    raise Exception(
        'model state dict "{}" does not exist'.format(filename))

  ckpt = tf.train.Checkpoint(
      decoder=decoder, epoch=tf.Variable(0, dtype=tf.int64))
  ckpt.restore(filename)
  epoch = ckpt.epoch

  return epoch


def build_decoder(experiment_directory, experiment_specs):

  arch = __import__(
      "networks." + experiment_specs["NetworkArch"], fromlist=["Decoder"]
  )

  latent_size = experiment_specs["CodeLength"]

  decoder = arch.Decoder(
      latent_size, **experiment_specs["NetworkSpecs"])

  return decoder


def load_decoder(
    experiment_directory, experiment_specs, checkpoint, data_parallel=True
):

  decoder = build_decoder(experiment_directory, experiment_specs)
  epoch = load_model_parameters(experiment_directory, checkpoint, decoder)

  return (decoder, epoch)


def get_data_source_map_filename(data_dir):
  return os.path.join(data_dir, data_source_map_filename)


def get_reconstructed_mesh_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

  return os.path.join(
      experiment_dir,
      reconstructions_subdir,
      str(epoch),
      reconstruction_meshes_subdir,
      dataset,
      class_name,
      instance_name + ".ply",
  )


def get_reconstructed_code_filename(
    experiment_dir, epoch, dataset, class_name, instance_name
):

  return os.path.join(
      experiment_dir,
      reconstructions_subdir,
      str(epoch),
      reconstruction_codes_subdir,
      dataset,
      class_name,
      instance_name + ".ckpt",
  )


def get_evaluation_dir(experiment_dir, checkpoint, create_if_nonexistent=False):

  dir = os.path.join(experiment_dir, evaluation_subdir, checkpoint)

  if create_if_nonexistent and not os.path.isdir(dir):
    os.makedirs(dir)

  return dir


def get_model_params_dir(experiment_dir, create_if_nonexistent=False):

  dir = os.path.join(experiment_dir, model_params_subdir)

  if create_if_nonexistent and not os.path.isdir(dir):
    os.makedirs(dir)

  return dir


def get_optimizer_params_dir(experiment_dir, create_if_nonexistent=False):

  dir = os.path.join(experiment_dir, optimizer_params_subdir)

  if create_if_nonexistent and not os.path.isdir(dir):
    os.makedirs(dir)

  return dir


def get_latent_codes_dir(experiment_dir, create_if_nonexistent=False):

  dir = os.path.join(experiment_dir, latent_codes_subdir)

  if create_if_nonexistent and not os.path.isdir(dir):
    os.makedirs(dir)

  return dir


def get_normalization_params_filename(
    data_dir, dataset_name, class_name, instance_name
):
  return os.path.join(
      data_dir,
      normalization_param_subdir,
      dataset_name,
      class_name,
      instance_name + ".npz",
  )
