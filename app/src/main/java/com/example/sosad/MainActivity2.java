package com.example.sosad;

import androidx.appcompat.app.AppCompatActivity;


import static androidx.constraintlayout.helper.widget.MotionEffect.TAG;


import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import android.speech.tts.TextToSpeech;
import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.TextUtils;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.sosad.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.PriorityQueue;
import java.util.Arrays;

public class MainActivity2 extends AppCompatActivity {


    TextView result, confidence;
    ImageView imageView;
    Button picture, selectBtn;
    TextToSpeech textToSpeech; // ตัวแปร TextToSpeech
    int imageSize = 224;


    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main2);

        result = findViewById(R.id.result);
        confidence = findViewById(R.id.confidence);
        imageView = findViewById(R.id.imageView);
        picture = findViewById(R.id.button);
        selectBtn = findViewById(R.id.selectBtn);

        picture.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
                // Check for camera permission
                if (ContextCompat.checkSelfPermission(MainActivity2.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                    openCamera();
                } else {
                    // Request camera permission if not granted
                    ActivityCompat.requestPermissions(MainActivity2.this, new String[]{Manifest.permission.CAMERA}, 100);
                }
            }
        });

        selectBtn.setOnClickListener(new View.OnClickListener() {
            public void onClick(View view) {
                openGallery();
            }
        });

        // ตรวจสอบการอนุญาตใช้งาน RECORD_AUDIO
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, 1); // ใช้ 1 หรือค่าคงที่อื่น ๆ ที่คุณต้องการ
        }

        textToSpeech = new TextToSpeech(this, new TextToSpeech.OnInitListener() {

            public void onInit(int status) {
                if (status == TextToSpeech.SUCCESS) {
                    int result = textToSpeech.setLanguage(Locale.US); // ตั้งค่าภาษา
                    if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                        Log.e(TAG, "Language not supported");
                    } else {
                        // TTS พร้อมใช้งาน ตอนนี้คุณสามารถเรียก speakText() ได้
                    }
                } else {
                    Log.e(TAG, "TTS initialization failed");
                }
            }
        });

    }

    private static final int PICK_IMAGE_REQUEST = 1;

    // Open the gallery
    private void openGallery() {
        Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(galleryIntent, PICK_IMAGE_REQUEST);
    }

    // Open the camera
    private void openCamera() {
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(cameraIntent, 1);
    }

    private void speakText(String text) {
        if (textToSpeech != null) {
            textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, null, null);
        }
    }

    @Override
    protected void onDestroy() {
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
        super.onDestroy();
    }


    public void classifyImage(Bitmap image) {
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Create inputs for the model.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());
            int pixel = 0;
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f / 255.f));
                    byteBuffer.putFloat((val & 0xFF) * (1.f / 255.f));
                }
            }

            inputFeature0.loadBuffer(byteBuffer);

            // Run model inference and get the result.

            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            model.close();

            float[] confidences = outputFeature0.getFloatArray();

            List<Integer> topClasses = findTopNClasses(outputFeature0.getFloatArray(), 3);

            String[] classes = {"Airedale", "American Staffordshire Terrier", "Appenzeller", "Basenji", "Beagle", "Bernese Mountain Dog", "Blenheim spaniel", "Bluetick", "Boston Bull", "Boxer",
                    "Brabancon Griffon", "Bull Mastiff", "Chihuahua", "Chow", "Cocker Spaniel", "Collie", "Dingo", "Doberman", "English Foxhound", "EntleBucher",
                    "Flat Coated Retriever", "French Bulldog", "German Shepherd", "Giant Schnauzer", "Golden Retriever", "Gordon Setter", "Great Dane", "Great Pyrenees", "Greater Swiss Mountain Dog", "Groenendael",
                    "Ibizan Hound", "Irish Setter", "Irish Terrier", "Irish Water Spaniel", "Irish Wolfhound", "Italian Greyhound", "Japanese spaniel", "Keeshond", "Kelpie", "Kerry Blue Terrier",
                    "Komondor", "Kuvasz", "Labrador Retriever", "Lakeland Terrier", "Leonberg", "Lhasa", "Malamute", "Malinois", "Maltese dog", "Mexican Hairless",
                    "Miniature Pinscher", "Miniature Poodle", "Miniature Schnauzer", "Newfoundland", "Norfolk Terrier", "Norwegian Elkhound", "Norwich Terrier", "Pekinese", "Pomeranian", "Pug",
                    "Redbone", "Rhodesian Ridgeback", "Rottweiler", "Saint Bernard", "Saluki", "Samoyed", "Shihtzu", "Siberian Husky", "Silky Terrier", "Staffordshire Bullterrier",
                    "Standard Poodle", "Standard Schnauzer", "Toy Poodle", "Vizsla", "Walker Hound", "Weimaraner", "West Highland White Terrier", "Wire Haired Fox Terrier", "Yorkshire Terrier"};


            // Sort topClasses in descending order of confidence scores
            topClasses.sort(new Comparator<Integer>() {
                public
                int compare(Integer classIndex1, Integer classIndex2) {
                    return Float.compare(confidences[classIndex2], confidences[classIndex1]);
                }
            });

            // Create an array to store the top 3 classes and their confidence scores
            String[] top3Classes = new String[Math.min(3, topClasses.size())];

            // Populate the top3Classes array with class names and confidence scores
            for (int i = 0; i < top3Classes.length; i++) {
                int classIndex = topClasses.get(i);
                String className = classes[classIndex];
                float confidenceScore = confidences[classIndex];

                // Add a condition to check for valid confidence scores
                if (!Float.isNaN(confidenceScore)) {
                    top3Classes[i] = className + ": " + String.format("%.1f%%", confidenceScore * 100);
                }
            }

            // Sort top3Classes array in descending order of confidence scores
            Arrays.sort(top3Classes, new Comparator<String>() {
                @Override
                public int compare(String s1, String s2) {
                    float confidence1 = Float.parseFloat(s1.split(": ")[1].replace("%", ""));
                    float confidence2 = Float.parseFloat(s2.split(": ")[1].replace("%", ""));
                    return Float.compare(confidence2, confidence1);
                }
            });

// นำข้อมูลคลาสที่มีความมั่นใจสูงสุดมาแสดงใน TextView confidence
            String topConfidenceClass = top3Classes[0]; // คลาสที่มีความมั่นใจสูงสุด

// ตั้งค่าให้ TextView confidence แสดงคลาสที่มีความมั่นใจสูงสุด
            confidence.setText(topConfidenceClass);



            result.setText(classes[topClasses.get(0)]); // Display the top class in TextView result
            confidence.setText(TextUtils.join("\n", top3Classes)); // Display the top 3 classes in TextView confidence

            speakText(result.getText().toString());
        } catch (IOException e) {
            // Handle the exception
        }
    }

    private List<Integer> findTopNClasses(float[] predictions, int n) {
        List<Integer> topClasses = new ArrayList<>();
        PriorityQueue<Pair<Integer, Float>> queue = new PriorityQueue<>(n, (a, b) -> Float.compare(b.second, a.second)); // จัดลำดับตามความมั่นใจจากมากไปน้อย

        for (int i = 0; i < predictions.length; i++) {
            Pair<Integer, Float> pair = new Pair<>(i, predictions[i]);
            if (queue.size() < n) {
                queue.add(pair);
            } else {
                if (predictions[i] > queue.peek().second) {
                    queue.poll();
                    queue.add(pair);
                }
            }
        }

        while (!queue.isEmpty()) {
            topClasses.add(queue.poll().first);
        }

        return topClasses;
    }


    // Class to store class index and confidence score
    private static
    class ClassPrediction {
        private int classIndex;
        private float confidence;

        public
        ClassPrediction(int classIndex, float confidence) {
            this.classIndex = classIndex;
            this.confidence = confidence;
        }
        public
        int getClassIndex() {
            return classIndex;
        }
        public
        float getConfidence() {
            return confidence;
        }
    }

    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == PICK_IMAGE_REQUEST && resultCode == RESULT_OK && data != null && data.getData() != null) {
            try {
                // เรียกใช้งานเมธอดที่คุณสร้างเพื่อดึงรูปภาพที่เลือก
                Bitmap selectedImage = MediaStore.Images.Media.getBitmap(this.getContentResolver(), data.getData());

                // ปรับขนาดรูปภาพและเรียกเมธอด classifyImage เพื่อทำนาย
                int dimension = Math.min(selectedImage.getWidth(), selectedImage.getHeight());
                selectedImage = ThumbnailUtils.extractThumbnail(selectedImage, dimension, dimension);

                // ตรวจสอบว่า ImageView ถูกตั้งค่าอย่างถูกต้อง
                if (imageView != null) {
                    imageView.setImageBitmap(selectedImage);
                } else {
                    Log.e(TAG, "ImageView is null");
                }

                selectedImage = Bitmap.createScaledBitmap(selectedImage, imageSize, imageSize, false);
                classifyImage(selectedImage);
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else if (requestCode == 1 && resultCode == RESULT_OK && data != null) {
            // เมื่อคุณถ่ายรูปจากกล้อง
            Bundle extras = data.getExtras();
            assert extras != null;
            Bitmap imageBitmap = (Bitmap) extras.get("data");

            // ปรับขนาดรูปภาพและแสดงบน ImageView
            if (imageView != null) {
                imageView.setImageBitmap(imageBitmap);
            }

            imageBitmap = Bitmap.createScaledBitmap(imageBitmap, imageSize, imageSize, false);

            // เรียกเมธอด classifyImage เพื่อทำนาย
            classifyImage(imageBitmap);
        }
    }

}

