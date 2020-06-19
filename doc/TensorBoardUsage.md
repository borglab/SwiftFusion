How to use the TensorBoard logging system
================

Currently the TensorBoard metric logging infra is only implemented for Pose3SLAMG2O demo application.

## Usage

To run an experiment, do the following:
```bash
mkdir -p logs/lm_sphere # Your experiment name
swift build -c release -Xswiftc -cross-module-optimization
swift run -c release -Xswiftc -cross-module-optimization Pose3SLAMG2O <g2o_file> ./<result_file>.txt -l ./logs/<your expr name>
```
At the same time, in a separate terminal:
```
# in a separate terminal
tensorboard dev upload --logdir logs \
    --name "My first SwiftFusion in tensorboard.dev" \
    --description "One small step for myself, a giant step for human kind."
```
There should be a logging prompt, log in to your google account.

Now there should be a prompt at the second terminal that says:
```
View your TensorBoard live at: https://tensorboard.dev/experiment/<xxxxx>/
```

Navigate to the URL on your web browser, and you should see the metric.
