## Working examples:
* Using runner sif
```bash
singularity run --nv --bind $HOME/path/arxivDownload:/opt/arxivDownload,/media/hd1:/opt/data_dir \
   $$HOME/singul/runner.sif python3 embed/classify_lstm.py 
   --model /opt/data_dir/trained_models/lstm_classifier/lstm_Aug-19_17-22 
   --out $HOME/path/rm_me/mine_inference 
   --mine /opt/data_dir/promath/math96
   ```
