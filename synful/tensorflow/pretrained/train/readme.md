##  Quick glance: Pretrained model specs


| Setup                     | Specs                           | F-score with seg | F-score without | Remarks|
|---------------------------|---------------------------------|------------------|------------------|------------------|
| p`_`setup52 (`+`p_setup10)    | big, curriculum, CE, ST          | 0.76             | 0.74             | **Highest Accuracy** |
| p`_`setup51                   | big, curriculum, CE, MT_2       | 0.76             | 0.73             |
| p`_`setup54 (`+`p_setup05)    | small, curriculum, MSE, ST       | 0.76             | 0.7              | **Fast inference, reasonable results** |
| p`_`setup45 (`+`p_setup05)    | small, standard, MSE, MT2        | 0.73             | 0.68             |

Note, that for the models that have an underlying ST architecture we also indicate the setup for the corresponding direction-vector-models (`p_setup05+p_setup10`). If you want to use the model with highest accuracy, pick `p_setup52`. If you want to use a model that gives reasonnable results, but also has fast inference runtime, pick `p_setup54`.


## Download Checkpoints
Checkpoints: [Link](https://www.dropbox.com/scl/fo/hlw1cbef09xwisss59fhr/h?rlkey=uk7786539u1fu21dh5ebowip6&dl=0)


