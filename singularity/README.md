## Working examples:
* Using runner.sif singularity container to classify paragraphs [master 242d718]
```bash
singularity run --nv --bind $HOME/path/arxivDownload:/opt/arxivDownload,/media/hd1:/opt/data_dir \
   $$HOME/singul/runner.sif python3 embed/classify_lstm.py 
   --model /opt/data_dir/trained_models/lstm_classifier/lstm_Aug-19_17-22 
   --out $HOME/path/rm_me/mine_inference 
   --mine /opt/data_dir/promath/math96
   ```

```bash
singularity run --nv --bind $HOME/arxivDownload:/opt/arxivDownload,$PROJECT:/opt/data_dir 
    $HOME/singul/runner.sif python3 embed/classify_lstm.py 
    --model /opt/data_dir/trained_models/lstm_classifier/lstm_Feb-22_22-51 
    --out $HOME/rm_me/mine_inference 
    --mine /opt/data_dir/promath/math96
```

