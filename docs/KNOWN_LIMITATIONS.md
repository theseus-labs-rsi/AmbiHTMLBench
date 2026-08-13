# Known limitations

- The VLM Judge is not equivalent to a human reviewer. Final Arena90 showed about 19.3% orientation disagreement; the release records and neutralizes this instability rather than claiming it is solved.
- A `reliability` TIE is not perceptual indistinguishability. It means canonical and swapped observations did not reproduce the same direction.
- Disagreement-to-TIE can sacrifice statistical power by neutralizing some pairs for which a human reviewer may still see a clear winner.
- The seven visual axes depend on the current GT naming conventions. A new annotation taxonomy requires an explicit mapping update and validation.
- Chromium/Playwright rendering cannot guarantee identical remote third-party assets when source HTML depends on unstable external resources. Fallbacks are bounded and recorded, but missing remote behavior cannot always be reconstructed.
- The final prompt comparison used one blind reviewer on 30 random pairs. It did not show prompt-level human-accuracy superiority. A multi-annotator study would be required for publication-level human inter-rater and Judge-human agreement estimates.
