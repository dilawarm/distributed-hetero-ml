FROM python:3.13-slim

WORKDIR "/python_template"
COPY . /python_template

ENV UV_PYTHON_PREFERENCE=managed
ENV UV_PYTHON=3.13
ENV UV_NO_PROGRESS=true

ENV PATH="/root/.local/bin:${PATH}"
ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt-get update -yq
RUN env DEBIAN_FRONTEND=noninteractive apt-get install -q -o=Dpkg::Use-Pty=0 -y --no-install-recommends pkg-config libicu-dev curl git make
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN pip install setuptools_scm
RUN uv run make install
RUN apt-get autoremove --purge -y && apt-get clean
RUN uv clean
