# syntax=docker/dockerfile:1.4

FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/root/.local/bin:$PATH

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked <<-EOT
  apt-get update
  apt-get install -y python3-pip libglib2.0-0
EOT

COPY requirement.txt /agentformer/requirement.txt

RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked <<-EOT
  pip3 install --user --upgrade pip
  pip3 install -r /agentformer/requirement.txt
EOT
