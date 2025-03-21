# Start with the latest Ubuntu image
FROM ubuntu:latest as base

# Prevent interactive prompts and install OS-level dependencies
# (Add or remove packages as needed for your project)
RUN apt-get update && apt-get install -y \
    python3-pip python3 python3-venv \
    git screen libiio-dev libiio-utils vim python3-dev uhubctl libusb-dev libusb-1.0-0-dev \
    && rm -rf /var/lib/apt/lists/*

# Clone your repo
RUN git clone https://github.com/misko/spf.git /spf

# Change working directory
WORKDIR /spf

RUN python3 -m venv /opt/spf-env
ENV PATH="/opt/spf-env/bin:$PATH"

COPY .env /app/.env

RUN pip install --upgrade pip
# In this stage, you can confirm everything's installed

# Install your package (which has a pyproject.toml)
# If you have a specific dev or optional dependencies, you can add them here as well
RUN pip install -e .

# Launch your tests; you can change this to 'pytest' or any other test runner
WORKDIR /app/
#config=..//model_training_and_inference/models/ekf_and_pf_config.yml
ENV CONFIG=/spf/spf/model_training_and_inference/models/ekf_and_pf_config.yml 
#ENV PKLFN=b2://projectspf/md2/cache/inference_models/paired_checkpoints_windowedbeamformer_nosig_3p5_randx_wd0p02_gains_vehicle_0p2dropout_noroverbounce/empirical_dist.pkl
ENV PKLFN=b2://projectspf/md2/cache/inference_models/checkpoints/march16/paired_wd0p02_gains_vehicle_0p2dropout_noroverbounceREAL_lowdrop_x2/empirical_dist.pkl
ENV NAME=march16inference
#ENV MANIFEST=b2://projectspf/md2/splits/feb16_val_nosig_with_rover.txt 
ENV MANIFEST=b2://projectspf/md2/splits/march16_val_nosig_noroverbounce.txt 
CMD /opt/spf-env/bin/python3 /spf/spf/scripts/run_filters_on_data_b2.py --config "$CONFIG" --empirical-pkl-fn "$PKLFN"  --parallel 28 --work-dir "$NAME" --manifest "$MANIFEST"

#CMD ["pytest", "tests"]
