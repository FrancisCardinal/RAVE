#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <algorithm>

const int BUFFER_SIZE = 256;
int ADMAIFInterface = 1;
int I2SInterface = 1;
int channelNumber = 8;
int frequency = 48000;
int duration = 4;             // secs
std::string fileType = "wav"; // raw or wav

void system_no_output(const char *cmd)
{
  std::string s = std::string(cmd) + " > nul 2> nul";
  system(s.c_str());
}

void initEnvironment()
{
  std::string c1 = "amixer -c tegrasndt186ref cset name=\"ADMAIF" + std::to_string(ADMAIFInterface) + " Mux\" \"I2S" + std::to_string(I2SInterface) + "\"";
  std::string c2 = "amixer -c tegrasndt186ref cset name=\"ADMAIF" + std::to_string(ADMAIFInterface) + " Channels\" " + std::to_string(channelNumber);
  std::string c3 = "amixer -c tegrasndt186ref cset name=\"I2S" + std::to_string(I2SInterface) + " Channels\" " + std::to_string(channelNumber);
  std::string c4 = "amixer -c tegrasndt186ref cset name=\"I2S" + std::to_string(I2SInterface) + " Sample Rate\" " + std::to_string(frequency);
  std::string c5 = "amixer -c tegrasndt186ref cset name=\"I2S" + std::to_string(I2SInterface) + " codec bit format\" 32";
  std::string c6 = "amixer -c tegrasndt186ref cset name=\"I2S" + std::to_string(I2SInterface) + " input bit format\" 32";
  std::string c7 = "amixer -c tegrasndt186ref cset name=\"I2S" + std::to_string(I2SInterface) + " codec master mode\" \"cbs-cfs\"";
  std::string c8 = "amixer -c tegrasndt186ref cset name=\"I2S" + std::to_string(I2SInterface) + " codec frame mode\" \"dsp-a\"";
  std::string c9 = "amixer -c tegrasndt186ref cset name=\"I2S" + std::to_string(I2SInterface) + " fsync width\" 0";

  system_no_output(c1.c_str());
  system_no_output(c2.c_str());
  system_no_output(c3.c_str());
  system_no_output(c4.c_str());
  system_no_output(c5.c_str());
  system_no_output(c6.c_str());
  system_no_output(c7.c_str());
  system_no_output(c8.c_str());
  system_no_output(c9.c_str());
  // printf(c1.c_str());
  // printf("\n");
  // printf(c2.c_str());
  // printf("\n");
  // printf(c3.c_str());
  // printf("\n");
  // printf(c4.c_str());
  // printf("\n");
  // printf(c5.c_str());
  // printf("\n");
  // printf(c6.c_str());
  // printf("\n");
  // printf(c7.c_str());
  // printf("\n");
  // printf(c8.c_str());
  // printf("\n");
  // printf(c9.c_str());
  // printf("\n");
}

// Flips a byte array byte by byte
void flip(char *buffer, int sizeOfBuffer)
{
  for (int i = 0; i < sizeOfBuffer; i++)
  {
    char *start = &buffer[i];
    char *end = &buffer[i + 1];
    std::reverse(start, end);
  }
}

int main(int argc, char **argv)
{
  initEnvironment();
  std::string recordCommand = "arecord -D hw:tegrasndt186ref," + std::to_string(ADMAIFInterface - 1) + " -r " + std::to_string(frequency) + " -c " + std::to_string(channelNumber) + " -f S32_LE -d " + std::to_string(duration) + " -t " + fileType + " -q -";
  char buf[BUFFER_SIZE];
  FILE *fp = popen(recordCommand.c_str(), "r");
  std::string fileOutput = "./out." + fileType;
  FILE *out = fopen(fileOutput.c_str(), "wb");
  int result;
  // printf("Recording...\n");
  //  Do not flip the WAV file header, which is 44 bytes long
  if (fileType == "wav")
  {
    for (int i = 0; i < 44; i++)
    {
      result = fread(buf, 1, 1, fp);
      fwrite(buf, 1, 1, out);
    }
  }

  // While data is available
  while (result = fread(buf, 1, BUFFER_SIZE, fp))
  {
    // Flip bytes and write
    flip(buf, BUFFER_SIZE);
    fwrite(buf, 1, BUFFER_SIZE, out);
  }

  // printf("Done\n");
  pclose(fp);
  fclose(out);
  return 0;
}