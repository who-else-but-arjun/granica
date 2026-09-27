# Savor Segmentation Contract

MobileSAM receives the complete original plate photograph plus one OpenAI
grounded, native-pixel food box per menu item portion. Keep each output mask
inside its corresponding prompt box before counting pixels; never substitute a
color threshold or the whole compartment when a mask is empty. Store the mask
pixel count, the prompt/refined boxes, and the menu item label together. Render
the mask with a strong opaque color overlay and draw the item name next to its
box. Treat empty masks and suspiciously broad boxes as review flags rather than
silently expanding them.
