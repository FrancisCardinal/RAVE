#include <stdio.h>
int main(int argc, char **argv)
{
  char *cmd = "aplay -D plughw:0 -f S16_LE -c 1 -r 16000 -t raw -q -";
  char buf[256] = {
      0,
  };
  FILE *fp = popen(cmd, "r");
  for (int i = 0; i < 16; i++)
  {
    int result = fwrite(buf, 1, sizeof(buf), fp);
    printf("write %d bytes\n", result);
  }
  pclose(fp);
  return 0;
}