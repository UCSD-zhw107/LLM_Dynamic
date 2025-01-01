#!/bin/bash

# Check ENV Name
if [ $# -ne 1 ]; then
    echo "Usage: $0 <new_env_name>"
    exit 1
fi

ENV_NAME=$1
ISAAC_SIM_PATH="/home/zhw/.local/share/ov/pkg/isaac-sim-4.1.0"
GIBSON_SITE_PACKAGES="/home/zhw/.conda/envs/gibson/lib/python3.10/site-packages"

echo "Activating Conda environment: $ENV_NAME"

# Activate Env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME" || { echo "Failed to activate environment $ENV_NAME"; exit 1; }

# Add Gibson Site Package
echo "Adding Gibson site-packages to $ENV_NAME"
echo "$GIBSON_SITE_PACKAGES" > $(python -c "import site; print(site.getsitepackages()[0]")/gibson.pth

# Activate & Deactivate folder
ACTIVATE_D=$CONDA_PREFIX/etc/conda/activate.d
DEACTIVATE_D=$CONDA_PREFIX/etc/conda/deactivate.d

mkdir -p "$ACTIVATE_D" "$DEACTIVATE_D"

# ISSAC PATH for Activate
cat > "$ACTIVATE_D/env_vars.sh" <<EOL
#!/bin/sh
export ISAAC_PATH=$ISAAC_SIM_PATH
export PYTHONPATH=\$PYTHONPATH:$ISAAC_SIM_PATH/../../../:$ISAAC_SIM_PATH/kit/python/lib/python3.10/site-packages:$ISAAC_SIM_PATH/python_packages:$ISAAC_SIM_PATH/exts/omni.isaac.kit:$ISAAC_SIM_PATH/exts/omni.isaac.gym:$ISAAC_SIM_PATH/kit/kernel/py:$ISAAC_SIM_PATH/kit/plugins/bindings-python:$ISAAC_SIM_PATH/exts/omni.isaac.lula/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.exporter.urdf/pip_prebundle:$ISAAC_SIM_PATH/kit/exts/omni.kit.pip_archive/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.isaac.core_archive/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.isaac.ml_archive/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.pip.compute/pip_prebundle:$ISAAC_SIM_PATH/exts/omni.pip.cloud/pip_prebundle
export CARB_APP_PATH=$ISAAC_SIM_PATH/kit
export EXP_PATH=$ISAAC_SIM_PATH/apps
export LD_LIBRARY_PATH=$ISAAC_SIM_PATH/../../../:$ISAAC_SIM_PATH/.:$ISAAC_SIM_PATH/exts/omni.usd.schema.isaac/plugins/IsaacSensorSchema/lib:$ISAAC_SIM_PATH/exts/omni.usd.schema.isaac/plugins/RangeSensorSchema/lib:$ISAAC_SIM_PATH/kit:$ISAAC_SIM_PATH/kit/kernel/plugins:$ISAAC_SIM_PATH/kit/libs/iray:$ISAAC_SIM_PATH/kit/plugins:$ISAAC_SIM_PATH/kit/plugins/bindings-python:$ISAAC_SIM_PATH/kit/plugins/carb_gfx:$ISAAC_SIM_PATH/kit/plugins/rtx:$ISAAC_SIM_PATH/kit/plugins/gpu.foundation:\$LD_LIBRARY_PATH
EOL
chmod +x "$ACTIVATE_D/env_vars.sh"

# ISSAC PATH for Deactivate
cat > "$DEACTIVATE_D/env_vars.sh" <<EOL
#!/bin/sh
unset ISAAC_PATH
unset CARB_APP_PATH
unset EXP_PATH
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH_OLD}
unset LD_LIBRARY_PATH_OLD
export PYTHONPATH=\${PYTHONPATH_OLD}
unset PYTHONPATH_OLD
EOL
chmod +x "$DEACTIVATE_D/env_vars.sh"

echo "Setup complete! Activate the environment using 'conda activate $ENV_NAME' to load the Isaac Sim configuration."
conda deactivate
