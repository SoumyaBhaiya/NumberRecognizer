package com.example.numberrecognizerapp

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color.Companion.Black
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.util.concurrent.TimeUnit
import androidx.core.graphics.createBitmap

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    DrawingScreen()
                }
            }
        }
    }
}

data class Stroke(val start: Offset, val end: Offset)

@Composable
fun DrawingScreen() {
    var strokes by remember { mutableStateOf(listOf<Stroke>()) }
    var lastPoint by remember { mutableStateOf<Offset?>(null) }
    var prediction by remember { mutableStateOf("Prediction: ") }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.Center
    ) {
        Box(
            modifier = Modifier
                .size(300.dp)
                .background(androidx.compose.ui.graphics.Color.White)
        ) {
            Canvas(
                modifier = Modifier
                    .matchParentSize()
                    .pointerInput(Unit) {
                        detectDragGestures(
                            onDragStart = { offset -> lastPoint = offset },
                            onDragEnd = { lastPoint = null },
                            onDragCancel = { lastPoint = null },
                            onDrag = { change, _ ->
                                val prev = lastPoint
                                if (prev != null) {
                                    strokes = strokes + Stroke(prev, change.position)
                                }
                                lastPoint = change.position
                            }
                        )
                    }
            ) {
                for (s in strokes) {
                    drawLine(Black, s.start, s.end, strokeWidth = 20f)
                }
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Row(horizontalArrangement = Arrangement.spacedBy(16.dp)) {
            Button(onClick = { strokes = emptyList() }) {
                Text("Clear")
            }

            Button(onClick = {
                CoroutineScope(Dispatchers.IO).launch {
                    val result = sendToServer(strokes)
                    withContext(Dispatchers.Main) {
                        prediction = "Prediction: $result"
                    }
                }
            }) {
                Text("Predict")
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        Text(text = prediction, style = MaterialTheme.typography.headlineSmall)
    }
}

private fun sendToServer(strokes: List<Stroke>): String {
    // Convert strokes to Bitmap
    val bitmap = createBitmap(300, 300)
    val canvas = Canvas(bitmap)
    canvas.drawColor(Color.WHITE)
    val paint = Paint().apply {
        color = Color.BLACK
        style = Paint.Style.STROKE
        strokeWidth = 20f
    }

    for (s in strokes) {
        canvas.drawLine(s.start.x, s.start.y, s.end.x, s.end.y, paint)
    }

    val stream = ByteArrayOutputStream()
    bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream)
    val byteArray = stream.toByteArray()

    // HTTP client with timeout
    val client = OkHttpClient.Builder()
        .callTimeout(10, TimeUnit.SECONDS)
        .build()

    val body = MultipartBody.Builder().setType(MultipartBody.FORM)
        .addFormDataPart(
            "file", "drawing.png",
            RequestBody.create("image/png".toMediaTypeOrNull(), byteArray)
        )
        .build()

    val request = Request.Builder()
        .url("http://10.0.2.2:8000/predict") // emulator -> localhost:8000
        .post(body)
        .build()

    val response = client.newCall(request).execute()
    val json = JSONObject(response.body?.string() ?: "{}")
    return json.optString("prediction", "error")
}
