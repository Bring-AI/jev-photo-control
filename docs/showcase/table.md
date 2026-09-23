| Input | Prompt | What the model decided | Output |
|---|---|---|---|
| <img src="docs/showcase/cat.jpg" width="220"> | “blur the background” | blur everything but box [108, 304, 807, 717] | <img src="docs/showcase/cat-1.jpg" width="220"> |
| <img src="docs/showcase/cat.jpg" width="220"> | “crop tightly around the cat's face” | crop box [108, 361, 384, 716] | <img src="docs/showcase/cat-2.jpg" width="220"> |
| <img src="docs/showcase/cat.jpg" width="220"> | “make it black and white” | saturation ×0.0 | <img src="docs/showcase/cat-3.jpg" width="220"> |
| <img src="docs/showcase/taxi.jpg" width="220"> | *(empty: auto-enhance)* | contrast ×1.2 | <img src="docs/showcase/taxi-1.jpg" width="220"> |
| <img src="docs/showcase/taxi.jpg" width="220"> | “remove the orange color cast so it looks natural” | warmth -0.8 | <img src="docs/showcase/taxi-2.jpg" width="220"> |
| <img src="docs/showcase/taxi.jpg" width="220"> | “blur everything except the yellow taxi in front” | blur everything but box [442, 310, 1000, 1000] | <img src="docs/showcase/taxi-3.jpg" width="220"> |
| <img src="docs/showcase/lake.jpg" width="220"> | “make it warmer, like golden hour” | warmth +0.8 | <img src="docs/showcase/lake-1.jpg" width="220"> |
| <img src="docs/showcase/lake.jpg" width="220"> | “crop to the snowy mountain and its reflection” | crop box [158, 411, 411, 851] | <img src="docs/showcase/lake-2.jpg" width="220"> |
| <img src="docs/showcase/lake.jpg" width="220"> | “make the colors more vivid” | saturation ×1.3 | <img src="docs/showcase/lake-3.jpg" width="220"> |
| <img src="docs/showcase/beach.jpg" width="220"> | *(empty: auto-enhance)* | no edit | <img src="docs/showcase/beach-1.jpg" width="220"> |
| <img src="docs/showcase/beach.jpg" width="220"> | “darken the bright sky” | no edit | <img src="docs/showcase/beach-2.jpg" width="220"> |
| <img src="docs/showcase/beach.jpg" width="220"> | “rotate 3 degrees counter-clockwise” | rotate +3.0° | <img src="docs/showcase/beach-3.jpg" width="220"> |
