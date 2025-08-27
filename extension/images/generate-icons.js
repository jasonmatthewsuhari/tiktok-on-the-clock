import sharp from 'sharp';

const sizes = [16, 48, 128];

async function generateIcons() {
  for (const size of sizes) {
    await sharp('icon.svg')
      .resize(size, size)
      .png()
      .toFile(`icon${size}.png`);
    console.log(`Generated ${size}x${size} icon`);
  }
}

generateIcons().catch(console.error);
