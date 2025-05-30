


#include "dataArray.h"
#include "math.h"
#include <Chirale_TensorFlowLite.h>
// include static array definition of pre-trained model
#include "model.h"
// This TensorFlow Lite Micro Library for Arduino is not similar to standard
// Arduino libraries. These additional header files must be included.
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define N_SELECTED_FFT 200


double goertzel_complex(const double* signal, int k, int N) {
    
    double omega = 2.0 * M_PI * k / N;
    double cosine = cos(omega);
    double sine = sin(omega);
    double coeff = 2.0 * cosine;

    double q0 = 0.0, q1 = 0.0, q2 = 0.0;

    for (int i = 0; i < N; ++i) {
        q0 = coeff * q1 - q2 + signal[i];
        q2 = q1;
        q1 = q0;
    }

    // Compute the real and imaginary parts
    double real = q1 - q2 * cosine;
    double imag = q2 * sine;
    //out_v = {real, imag};
    
    return sqrt(pow(real,2) + pow(imag,2) );
}


// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

constexpr int kTensorArenaSize = 2000;
// Keep aligned to 16 bytes for CMSIS
alignas(16) uint8_t tensor_arena[kTensorArenaSize];
}  // namespace




void setup(){

  Serial.begin(9600);
  while(!Serial);

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
}

void loop()
{

  double suma = 0.0;
  double goert_abs;
  double feat_vector[N_SELECTED_FFT];
  TfLiteTensor* model_input = interpreter->input(0);

  unsigned long startTime;
  unsigned long endTime; // Record end time (ms)
  unsigned long fft_time, executionTime; // Calculate duration

  startTime = micros();

for (int i = 0; i < N_SELECTED_FFT; i++){
  //goert_abs = goertzel_complex(dataArray, xai_indexes[i], 10000); 
  feat_vector[i] = goertzel_complex(dataArray, xai_indexes[i], 10000)/5000; 
  //Serial.print("Computed fft coeff: ");
  //Serial.println(goert_abs/5000, 8);
}

// FFT TIME 
fft_time = micros() - startTime; // millis()

Serial.print("Execution time FFT (us): ");
Serial.print(fft_time);
Serial.print("\n");


for (int i = 0; i < N_SELECTED_FFT; ++i) {
      model_input->data.f[i] = feat_vector[i];
} 

TfLiteStatus invoke_status = interpreter->Invoke();
if (invoke_status != kTfLiteOk) {
      MicroPrintf("Invoke failed");
      return;
}

TfLiteTensor* output = interpreter->output(0);

endTime = micros();
executionTime = endTime - startTime;

Serial.print("\nANN outputs = \n");

Serial.print("Normal state probability: ");
Serial.print(output->data.f[0]*100,4);
Serial.print("\n");

Serial.print("Fault 1 state probability: ");
Serial.print(output->data.f[1]*100,4);
Serial.print("\n");

Serial.print("Fault 2 state probability: ");
Serial.print(output->data.f[2]*100,4);
Serial.print("\n");

Serial.print("Model inference time (us): ");
Serial.print(executionTime - fft_time);
Serial.print("\n");


Serial.print("Total execution time (us): ");
Serial.print(executionTime);
Serial.print("\n");

delay(10000);
Serial.print("\n");
}
