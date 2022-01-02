# Data Formats

File: `proc/jan1_proc.pk`
===

### Format
type: Dict[String, Dict]

Top level keys:
- groupnames: `['ClipViTB32', 'ClipViTB16', 'Resnet18-pretrained', 'Densenet121-pretrained', 'Resnet18', 'Densenet121']`
   
D[groupname]:

- 'x': TestAccs ndarray(50)
- 'xsoft': TestSoftAccs ndarray(50)
- 'probs': softmax probabilities. ndarray(NUM_LABELS, 50) 
- 'freqs': frequencies. ndarray(NUM_LABELS, 50) 
- 'fsmooth': gaussian-smoothed frequencues. ndarray(NUM_LABELS, 50) 


### Data Standadization
Data has been 'standardized': x-axis is evenly-spaced.
For 'fsmooth', y-axis has been gaussian-smoothed with sigma=2.0

The preprocessing is done in: `dist-shift/preetum/dec31_unified.ipynb`

### Plotting example

In: `dist-shift/preetum/jan1_proc_plot_demo.ipynb`