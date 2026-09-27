# Face Matching Contract

Build one normalized gallery embedding per person from every detected enrollment
image, retaining paths and detection metadata. Compare every target face against
every database face with cosine similarity, rank the full list, and report rank-1,
same-person versus different-person similarity, EER, and failure-to-detect cases.
Save a bar chart per target whose x-axis names the database face image and whose
y-axis is similarity, with the true target highlighted.
