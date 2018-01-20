# Train convolutional neural networks (CNNs) to classify 10 classes of images (CIFAR-10) using PyTorch.

## Dataset:
The CIFAR-10 dataset consists of 60,000 natural images in 10 classes with 50,000 training images and 10,000 testing images.
Each image is an 3-channel color (RGB) image of 32x32 pixels in size.
The classes are ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, ‘truck’.

## Results:
<div style="direction: ltr;">
<table style="direction: ltr; border-collapse: collapse; border: 1pt solid #A3A3A3;" border="1" cellspacing="0" cellpadding="0">
<tbody>
<tr>
<td style="vertical-align: top; width: .9708in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<ul style="margin-left: .3034in; direction: ltr; unicode-bidi: embed; margin-top: 0in; margin-bottom: 0in;">
<ol style="margin-left: 0in; direction: ltr; unicode-bidi: embed; margin-top: 0in; margin-bottom: 0in; font-family: Tahoma; font-size: 14.0pt; font-weight: bold; font-style: normal;" type="A">
<li style="margin-top: 0; margin-bottom: 0; vertical-align: middle; font-weight: bold;" value="19"><span style="font-family: Tahoma; font-size: 14.0pt; font-weight: bold; font-style: normal;">No</span></li>
</ol>
</ul>
</td>
<td style="vertical-align: top; width: 2.5118in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;"><span style="font-weight: bold;">Network Configuration</span></p>
</td>
<td style="vertical-align: top; width: 1.1888in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;"><span style="font-weight: bold;">Epochs</span></p>
</td>
<td style="vertical-align: top; width: 1.5729in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;"><span style="font-weight: bold;">Training Time</span></p>
</td>
<td style="vertical-align: top; width: 1.0229in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;"><span style="font-weight: bold;">Accuracy</span></p>
</td>
</tr>
<tr>
<td style="vertical-align: top; width: .893in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">1</p>
</td>
<td style="vertical-align: top; width: 2.5118in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">Network Configuration 1</p>
</td>
<td style="vertical-align: top; width: 1.1888in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">2 epochs</p>
</td>
<td style="vertical-align: top; width: 1.5534in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">12.6 min</p>
</td>
<td style="vertical-align: top; width: .9194in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">55 %</p>
</td>
</tr>
<tr>
<td style="vertical-align: top; width: .893in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">2</p>
</td>
<td style="vertical-align: top; width: 2.5312in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">Network Configuration 1_2</p>
</td>
<td style="vertical-align: top; width: 1.1888in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">50 epochs</p>
</td>
<td style="vertical-align: top; width: 1.5534in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">6.5 hr</p>
</td>
<td style="vertical-align: top; width: .9194in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">59 %</p>
</td>
</tr>
<tr>
<td style="vertical-align: top; width: .893in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">3</p>
</td>
<td style="vertical-align: top; width: 2.5118in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">Network Configuration 2</p>
</td>
<td style="vertical-align: top; width: 1.1888in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">2 epochs</p>
</td>
<td style="vertical-align: top; width: 1.5534in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">26.8 min</p>
</td>
<td style="vertical-align: top; width: .9194in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">59 %</p>
</td>
</tr>
<tr>
<td style="vertical-align: top; width: .893in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">4</p>
</td>
<td style="vertical-align: top; width: 2.5312in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">Network Configuration 2_2</p>
</td>
<td style="vertical-align: top; width: 1.1888in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">50 epochs</p>
</td>
<td style="vertical-align: top; width: 1.5534in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">6 hr</p>
</td>
<td style="vertical-align: top; width: .9194in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">67 %</p>
</td>
</tr>
<tr>
<td style="vertical-align: top; width: .893in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">5</p>
</td>
<td style="vertical-align: top; width: 2.5118in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">Network Configuration 3</p>
</td>
<td style="vertical-align: top; width: 1.1888in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">50 epochs</p>
</td>
<td style="vertical-align: top; width: 1.5534in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">14.14 hr</p>
</td>
<td style="vertical-align: top; width: .9194in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">72 %</p>
</td>
</tr>
<tr>
<td style="vertical-align: top; width: .893in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">6</p>
</td>
<td style="vertical-align: top; width: 2.5118in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">Network Configuration 4</p>
</td>
<td style="vertical-align: top; width: 1.1888in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">25 epochs</p>
</td>
<td style="vertical-align: top; width: 1.5534in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">15.55 hr</p>
</td>
<td style="vertical-align: top; width: .9194in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">74 %</p>
</td>
</tr>
<tr>
<td style="vertical-align: top; width: .893in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">7</p>
</td>
<td style="vertical-align: top; width: 2.5118in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">Network Configuration 5</p>
</td>
<td style="vertical-align: top; width: 1.1888in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">25 epochs</p>
</td>
<td style="vertical-align: top; width: 1.5534in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">28 hr</p>
</td>
<td style="vertical-align: top; width: .9194in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">80 %</p>
</td>
</tr>
<tr>
<td style="vertical-align: top; width: .893in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">8</p>
</td>
<td style="vertical-align: top; width: 2.5312in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">Network Configuration 5_1</p>
</td>
<td style="vertical-align: top; width: 1.1888in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">50 epochs</p>
</td>
<td style="vertical-align: top; width: 1.5534in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">60.8 hr</p>
</td>
<td style="vertical-align: top; width: .9194in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">80 %</p>
</td>
</tr>
<tr>
<td style="vertical-align: top; width: .893in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">9</p>
</td>
<td style="vertical-align: top; width: 2.5312in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">Network Configuration 5_2</p>
</td>
<td style="vertical-align: top; width: 1.2083in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">100 epochs</p>
</td>
<td style="vertical-align: top; width: 1.5534in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">132 hr</p>
</td>
<td style="vertical-align: top; width: .9194in; padding: 4pt 4pt 4pt 4pt; border: 1pt solid #A3A3A3;">
<p style="margin: 0in; font-family: Tahoma; font-size: 14.0pt;">54 %</p>
</td>
</tr>
</tbody>
</table>
</div>

## Network Configuration 5

<img src="NetworkConfiguration_5.png" align="middle">

