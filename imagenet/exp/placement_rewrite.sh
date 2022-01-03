set -e
node='67'
{
bash exp/speed/slurm_run_auto_rewrite_0.sh 1 GoogLeNet '1' $node 513 True
bash exp/speed/slurm_run_auto_rewrite_0.sh 1 GoogLeNet '1' $node 513 False
} 

{
bash exp/speed/slurm_run_auto_rewrite_0.sh 1 ResNet18 '1' $node 1024 True
bash exp/speed/slurm_run_auto_rewrite_0.sh 1 ResNet18 '1' $node 1024 False
} 

{
bash exp/speed/slurm_run_auto_rewrite_0.sh 1 VGG19 '1' $node 1024 True
bash exp/speed/slurm_run_auto_rewrite_0.sh 1 VGG19 '1' $node 1024 False
} 

{
bash exp/speed/slurm_run_auto_rewrite_0.sh 1 MobileNetV2 '1' $node 1024 True
bash exp/speed/slurm_run_auto_rewrite_0.sh 1 MobileNetV2 '1' $node 1024 False
}


{
bash exp/speed/slurm_run_auto_rewrite_0.sh 1 ResNet50 '1' $node 513 True
bash exp/speed/slurm_run_auto_rewrite_0.sh 1 ResNet50 '1' $node 513 False 
}
