# Why does my segmentations not overlap with my RAW EM in Napari ?

<table>
  <tr>
    <td align="center">
      <h3>Mismatch</h3>
      <img width="851" height="701" alt="image" src="https://github.com/user-attachments/assets/5ad4d7c7-9649-4ed5-b7aa-455151bff0e2" />
    </td>
    <td align="center">
      <h3>Matched</h3>
      <img width="851" height="701" alt="image" src="https://github.com/user-attachments/assets/c91b1b93-adb6-4934-b605-bebfb74cf57f" />
    </td>
  </tr>
</table>

During training Local Shape Descriptors we use valid convolutions, which makes the shape of the output affinities smaller than the input raw.
It cuts off the border regions.
For example: if your raw shape is `(626, 1393, 1696)`, the output segmentation may be of shape `(502, 1269, 1572)`. So you can see that
the segmentation is missing about `124` z slices and `124` pixels on XY. So technically your segmentaton lies within `62:626-62,62:1393-62,62:1696-62`.
It crops equally for every direction.

**These are the series of checks you should do:**
- Check the shapes of the raw and segmentation. See if they follow shape ZYX orientation.
  ```python
      raw.shape
      Out[29]: (626, 1393, 1696)
      seg.shape
      Out[30]: (502, 1269, 1572)
  ```
- If you see that ZYX orientation does not match say raw is `(626, 1393, 1696)` and seg is `(1572, 1269, 502)`, you must transpose the segmentation.
  ```python
    seg_transposed = seg.T
  ```
- Now follow the script below to pad the seg to match the raw shape.
  ```python
    import numpy as np

    # Assuming seg and raw are your loaded numpy arrays
    # Example shapes:
    # seg.shape -> (502, 1269, 1572)
    # raw.shape -> (626, 1393, 1696)
    
    # Get the current and target shapes
    current_shape = seg.shape
    target_shape = raw.shape
    
    # Calculate the total padding needed for each axis
    pad_needed = [(target_shape[i] - current_shape[i]) for i in range(3)]
    
    # Distribute the padding before and after each axis to center the data
    # This handles both even and odd padding amounts correctly
    pad_width = [
        (pad // 2, pad - (pad // 2)) for pad in pad_needed
    ]
    
    # Apply the padding with a constant value of 0
    padded_array = np.pad(
        seg, 
        pad_width=pad_width, 
        mode='constant', 
        constant_values=0
    )
    
    # Verify the new shape
    print(f"Original shape: {seg.shape}")
    print(f"Target shape:   {raw.shape}")
    print(f"Padded shape:   {padded_array.shape}")

  ```
