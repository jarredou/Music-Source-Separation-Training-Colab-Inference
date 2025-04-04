[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Q5Q811R5YI)  
# Colab inference for ZFTurbo's [Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training/)

Separation : [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jarredou/Music-Source-Separation-Training-Colab-Inference/blob/main/Music_Source_Separation_Training_(Colab_Inference).ipynb)  

Custom Model Import Version [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jarredou/Music-Source-Separation-Training-Colab-Inference/blob/main/Music_Source_Separation_Training_(Colab_Inference)_CustomModel.ipynb)  

Manual ensemble tool : [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jarredou/Music-Source-Separation-Training-Colab-Inference/blob/main/Manual_Ensemble_Colab.ipynb)  
<br>  
<hr>  
<b>New notebook added to ease the use of custom models ! NEW !</b><br>
You can find a list of models here (maintained by Bas Curtiz):<br>
https://bascurtiz.x10.mx/models-checkpoint-config-urls.html or<br>
https://github.com/SiftedSand/MusicSepGUI/blob/main/models.json (by SeinfeldMaster)<br>
<br><br>
Available models:<br>
<b>"INST-Mel-Roformer v1e+ (by unwa)" NEW ! </b><br>
<b>"INST-Mel-Roformer v1+ Preview (by unwa)" NEW !</b><br>
<b>"INST-Mel-Roformer Metal Model Preview (by Mesk)" NEW !</b><br>
<b>"VOCALS-Mel-Roformer FT3 Preview (by unwa)" NEW !</b><br>
<b>"VOCALS-Mel-Roformer Big Beta 6X (by unwa)" NEW !</b><br>
<b>BandIt_v2 multi model (by kwatcharasupat)</b><br>
"INST-MelBand-Roformer INSTV7 (by Gabox)"<br>
"INST-MelBand-Roformer INSTFVX (by Gabox)"<br>
"INST-MelBand-Roformer INSTV7N (by Gabox)"<br>
"INST-Mel-Roformer gabox_inst3 (by Gabox)"<br>
"INST-Mel-Roformer Kim FT2 bleedless (by unwa)"<br>
"VOCALS-MelBand-Roformer Big Beta 6 (by unwa)"<br>
"INST-MelBand-Roformer INSTV6N (by Gabox)"<br>
"VOCALS-MelBand-Roformer voc_Fv4 (by Gabox)"<br>
"VOCALS-MelBand-Roformer voc_Fv3 (by Gabox)"<br>
"INST-Mel-Roformer INSTV6 (by Gabox)"<br>
"INST-Mel-Roformer INSTV5 (by Gabox)<br>
"VOCALS-Male-Female-Mel-Roformer 7.2889 FT (by Aufr33)"<br>
"VOCALS-Mel-Roformer Kim FT 2 (by unwa)"<br>
"INST-Mel-Roformer (by Becruily)"<br>
"VOCAL-Mel-Roformer (by Becruily)"<br>
"DEBLEED-Mel-Roformer (by unwa/97chris)"<br>
"4STEMS-SCNet-XL-MUSDB18 (by ZFTurbo)"<br>
"4STEMS-BS-Roformer-MUSDB18 (by ZFTurbo)"<br>
"4STEMS-SCNet-large (by starrytong)"<br>
"VOCALS-Mel-Roformer Kim FT (by unwa)"<br>
"4STEMS-SCNet-MUSDB18 (by starrytong)"<br>
"VOCALS-Mel-Roformer BigBeta5e (by unwa)"<br>
"INST-Mel-Roformer v1e (by unwa)"<br>
"INST-Mel-Roformer v2 (by unwa)"<br>
"INST-VOC-Mel-Roformer a.k.a. duality v2 (by unwa)"<br>
"INST-Mel-Roformer v1 (by unwa)"<br>
"INST-VOC-Mel-Roformer a.k.a. duality (by unwa)"<br>
"VOCALS-Mel-Roformer big beta 4 (by unwa)"<br>
"BS-Roformer Large V1 (by unwa)"<br>
"MDX23C DeReverb (by aufr33 and jarredou)"<br>
"MelBand-Roformer Karaoke (by aufr33 and viperx)"<br>
"Kim MelBand-vocals" (by KimberleyJSN)<br>
"MelBand-Denoise" (by aufr33)<br>
"MDX23C_InstVocHQ" (by Anjok, aufr33, ZFTurbo)<br> 
"BS-Roformer_1297" (by viperx)<br>
"BS-Roformer_1296" (by viperx)<br>
"BS-Roformer_1053" (by viperx)<br>
"SCNet_MUSDB18" (by starrytong)<br>
"MelBand-Roformer_Decrowd" (by aufr33)<br>
"VitLarge23" (by ZFTurbo)<br>
"BandIt_Plus" (by kwatcharasupat)<br>
"MDX23C_DrumSep_6stem" (by jarredou & aufr33)<br>
<br>
For becruily instrumental model consider using this  Colab:<br> https://colab.research.google.com/github/lucassantillifuck2fa/Music-Source-Separation-Training/blob/main/Phase_Fixer.ipynb with auto phase fix for results with less vocal residues.<br>
