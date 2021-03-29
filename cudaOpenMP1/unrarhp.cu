#include <omp.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "common.h"
#include "windowMigration.cpp"
#include <io.h>
#include <helper_cuda.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#define MAXPASSWD 128

#define _RAR_SHA1_

#define HW 5

typedef struct {
    uint32 state[5];
    uint32 count[2];
    unsigned char buffer[64];
} hash_context;

 /**************************************************************************
  * This code is based on Szymon Stefanek AES implementation:              *
  * http://www.esat.kuleuven.ac.be/~rijmen/rijndael/rijndael-cpplib.tar.gz *
  *                                                                        *
  * Dynamic tables generation is based on the Brian Gladman's work:        *
  * http://fp.gladman.plus.com/cryptography_technology/rijndael            *
  **************************************************************************/

#define _MAX_KEY_COLUMNS (256/32)
#define _MAX_ROUNDS      14
#define MAX_IV_SIZE      16
enum Direction { Encrypt, Decrypt };
struct Rijndael
{
    Direction m_direction;
    byte     m_initVector[MAX_IV_SIZE];
    byte     m_expandedKey[_MAX_ROUNDS + 1][4][4];
};

const int uKeyLenInBytes = 16, m_uRounds = 10;

__device__ static byte S[256], S5[256], rcon[30];
__device__ static byte T1[256][4], T2[256][4], T3[256][4], T4[256][4];
__device__ static byte T5[256][4], T6[256][4], T7[256][4], T8[256][4];
__device__ static byte U1[256][4], U2[256][4], U3[256][4], U4[256][4];


__device__ void Xor128(byte* dest, const byte* arg1, const byte* arg2)
{
#if defined(PRESENT_INT32) && defined(ALLOW_NOT_ALIGNED_INT)
    ((uint32*)dest)[0] = ((uint32*)arg1)[0] ^ ((uint32*)arg2)[0];
    ((uint32*)dest)[1] = ((uint32*)arg1)[1] ^ ((uint32*)arg2)[1];
    ((uint32*)dest)[2] = ((uint32*)arg1)[2] ^ ((uint32*)arg2)[2];
    ((uint32*)dest)[3] = ((uint32*)arg1)[3] ^ ((uint32*)arg2)[3];
#else
    for (int I = 0; I < 16; I++)
        dest[I] = arg1[I] ^ arg2[I];
#endif
}


__device__ void Xor128(byte* dest, const byte* arg1, const byte* arg2,
    const byte* arg3, const byte* arg4)
{
#if defined(PRESENT_INT32) && defined(ALLOW_NOT_ALIGNED_INT)
    (*(uint32*)dest) = (*(uint32*)arg1) ^ (*(uint32*)arg2) ^ (*(uint32*)arg3) ^ (*(uint32*)arg4);
#else
    for (int I = 0; I < 4; I++)
        dest[I] = arg1[I] ^ arg2[I] ^ arg3[I] ^ arg4[I];
#endif
}


__device__ void Copy128(byte* dest, const byte* src)
{
#if defined(PRESENT_INT32) && defined(ALLOW_NOT_ALIGNED_INT)
    ((uint32*)dest)[0] = ((uint32*)src)[0];
    ((uint32*)dest)[1] = ((uint32*)src)[1];
    ((uint32*)dest)[2] = ((uint32*)src)[2];
    ((uint32*)dest)[3] = ((uint32*)src)[3];
#else
    for (int I = 0; I < 16; I++)
        dest[I] = src[I];
#endif
}

#define ff_poly 0x011b
#define ff_hi   0x80

#define FFinv(x)    ((x) ? pow[255 - log[x]]: 0)

#define FFmul02(x) (x ? pow[log[x] + 0x19] : 0)
#define FFmul03(x) (x ? pow[log[x] + 0x01] : 0)
#define FFmul09(x) (x ? pow[log[x] + 0xc7] : 0)
#define FFmul0b(x) (x ? pow[log[x] + 0x68] : 0)
#define FFmul0d(x) (x ? pow[log[x] + 0xee] : 0)
#define FFmul0e(x) (x ? pow[log[x] + 0xdf] : 0)
#define fwd_affine(x) \
    (w = (uint)x, w ^= (w<<1)^(w<<2)^(w<<3)^(w<<4), (byte)(0x63^(w^(w>>8))))

#define inv_affine(x) \
    (w = (uint)x, w = (w<<1)^(w<<3)^(w<<6), (byte)(0x05^(w^(w>>8))))

__device__ void GenerateTables(struct Rijndael* covertedClass)
{
    unsigned char pow[512], log[256];
    int i = 0, w = 1;
    do
    {
        pow[i] = (byte)w;
        pow[i + 255] = (byte)w;
        log[w] = (byte)i++;
        w ^= (w << 1) ^ (w & ff_hi ? ff_poly : 0);
    } while (w != 1);

    for (int i = 0, w = 1; i < (int)sizeof(rcon) / (int)sizeof(rcon[0]); i++)
    {
        rcon[i] = w;
        w = (w << 1) ^ (w & ff_hi ? ff_poly : 0);
    }
    for (int i = 0; i < 256; ++i)
    {
        unsigned char b = S[i] = fwd_affine(FFinv((byte)i));
        T1[i][1] = T1[i][2] = T2[i][2] = T2[i][3] = T3[i][0] = T3[i][3] = T4[i][0] = T4[i][1] = b;
        T1[i][0] = T2[i][1] = T3[i][2] = T4[i][3] = FFmul02(b);
        T1[i][3] = T2[i][0] = T3[i][1] = T4[i][2] = FFmul03(b);
        S5[i] = b = FFinv(inv_affine((byte)i));
        U1[b][3] = U2[b][0] = U3[b][1] = U4[b][2] = T5[i][3] = T6[i][0] = T7[i][1] = T8[i][2] = FFmul0b(b);
        U1[b][1] = U2[b][2] = U3[b][3] = U4[b][0] = T5[i][1] = T6[i][2] = T7[i][3] = T8[i][0] = FFmul09(b);
        U1[b][2] = U2[b][3] = U3[b][0] = U4[b][1] = T5[i][2] = T6[i][3] = T7[i][0] = T8[i][1] = FFmul0d(b);
        U1[b][0] = U2[b][1] = U3[b][2] = U4[b][3] = T5[i][0] = T6[i][1] = T7[i][2] = T8[i][3] = FFmul0e(b);
    }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// API
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

__device__ void RijndaelInit(struct Rijndael* covertedClass)
{
    if (S[0] == 0)
        GenerateTables(covertedClass);
}

__device__ void keySched(struct Rijndael* covertedClass, byte key[_MAX_KEY_COLUMNS][4])
{
    int j, rconpointer = 0;

    // Calculate the necessary round keys
    // The number of calculations depends on keyBits and blockBits
    int uKeyColumns = m_uRounds - 6;

    byte tempKey[_MAX_KEY_COLUMNS][4];

    // Copy the input key to the temporary key matrix

    memcpy(tempKey, key, sizeof(tempKey));
    int r = 0;
    int t = 0;

    // copy values into round key array
    for (j = 0; (j < uKeyColumns) && (r <= m_uRounds); )
    {
        for (; (j < uKeyColumns) && (t < 4); j++, t++)
            for (int k = 0; k < 4; k++)
                covertedClass->m_expandedKey[r][t][k] = tempKey[j][k];
        if (t == 4)
        {
            r++;
            t = 0;
        }
    }

    while (r <= m_uRounds)
    {
        // Xor tempKey voi cac key ngay truoc khi
        tempKey[0][0] ^= S[tempKey[uKeyColumns - 1][1]];
        tempKey[0][1] ^= S[tempKey[uKeyColumns - 1][2]];
        tempKey[0][2] ^= S[tempKey[uKeyColumns - 1][3]];
        tempKey[0][3] ^= S[tempKey[uKeyColumns - 1][0]];
        tempKey[0][0] ^= rcon[rconpointer++];

        if (uKeyColumns != 8)
            for (j = 1; j < uKeyColumns; j++)
                for (int k = 0; k < 4; k++)
                    tempKey[j][k] ^= tempKey[j - 1][k];
        else
        {
            for (j = 1; j < uKeyColumns / 2; j++)
                for (int k = 0; k < 4; k++)
                    tempKey[j][k] ^= tempKey[j - 1][k];

            tempKey[uKeyColumns / 2][0] ^= S[tempKey[uKeyColumns / 2 - 1][0]];
            tempKey[uKeyColumns / 2][1] ^= S[tempKey[uKeyColumns / 2 - 1][1]];
            tempKey[uKeyColumns / 2][2] ^= S[tempKey[uKeyColumns / 2 - 1][2]];
            tempKey[uKeyColumns / 2][3] ^= S[tempKey[uKeyColumns / 2 - 1][3]];
            for (j = uKeyColumns / 2 + 1; j < uKeyColumns; j++)
                for (int k = 0; k < 4; k++)
                    tempKey[j][k] ^= tempKey[j - 1][k];
        }
        for (j = 0; (j < uKeyColumns) && (r <= m_uRounds); )
        {
            for (; (j < uKeyColumns) && (t < 4); j++, t++)
                for (int k = 0; k < 4; k++)
                    covertedClass->m_expandedKey[r][t][k] = tempKey[j][k];
            if (t == 4)
            {
                r++;
                t = 0;
            }
        }
    }
}

__device__ void keyEncToDec(struct Rijndael* covertedClass)
{
    for (int r = 1; r < m_uRounds; r++)
    {
        byte n_expandedKey[4][4];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
            {
                byte* w = covertedClass->m_expandedKey[r][j];
                n_expandedKey[j][i] = U1[w[0]][i] ^ U2[w[1]][i] ^ U3[w[2]][i] ^ U4[w[3]][i];
            }
        memcpy(covertedClass->m_expandedKey[r], n_expandedKey, sizeof(covertedClass->m_expandedKey[0]));
    }
}


__device__ void init(struct Rijndael* covertedClass, Direction dir, const byte* key, byte* initVector)
{
    covertedClass->m_direction = dir;

    byte keyMatrix[_MAX_KEY_COLUMNS][4];

    for (uint i = 0; i < (uint)uKeyLenInBytes; i++)
        keyMatrix[i >> 2][i & 3] = key[i];

    for (int i = 0; i < MAX_IV_SIZE; i++)
        covertedClass->m_initVector[i] = initVector[i];
    keySched(covertedClass, keyMatrix);
    if (covertedClass->m_direction == Decrypt)
        keyEncToDec(covertedClass);
}


__device__ void decrypt(struct Rijndael* covertedClass, const byte a[16], byte b[16])
{
    int r;
    byte temp[4][4];

    Xor128((byte*)temp, (byte*)a, (byte*)covertedClass->m_expandedKey[m_uRounds]);

    Xor128(b, T5[temp[0][0]], T6[temp[3][1]], T7[temp[2][2]], T8[temp[1][3]]);
    Xor128(b + 4, T5[temp[1][0]], T6[temp[0][1]], T7[temp[3][2]], T8[temp[2][3]]);
    Xor128(b + 8, T5[temp[2][0]], T6[temp[1][1]], T7[temp[0][2]], T8[temp[3][3]]);
    Xor128(b + 12, T5[temp[3][0]], T6[temp[2][1]], T7[temp[1][2]], T8[temp[0][3]]);


    for (r = m_uRounds - 1; r > 1; r--)
    {
        Xor128((byte*)temp, (byte*)b, (byte*)covertedClass->m_expandedKey[r]);
        Xor128(b, T5[temp[0][0]], T6[temp[3][1]], T7[temp[2][2]], T8[temp[1][3]]);
        Xor128(b + 4, T5[temp[1][0]], T6[temp[0][1]], T7[temp[3][2]], T8[temp[2][3]]);
        Xor128(b + 8, T5[temp[2][0]], T6[temp[1][1]], T7[temp[0][2]], T8[temp[3][3]]);
        Xor128(b + 12, T5[temp[3][0]], T6[temp[2][1]], T7[temp[1][2]], T8[temp[0][3]]);
    }

    Xor128((byte*)temp, (byte*)b, (byte*)covertedClass->m_expandedKey[1]);
    b[0] = S5[temp[0][0]];
    b[1] = S5[temp[3][1]];
    b[2] = S5[temp[2][2]];
    b[3] = S5[temp[1][3]];
    b[4] = S5[temp[1][0]];
    b[5] = S5[temp[0][1]];
    b[6] = S5[temp[3][2]];
    b[7] = S5[temp[2][3]];
    b[8] = S5[temp[2][0]];
    b[9] = S5[temp[1][1]];
    b[10] = S5[temp[0][2]];
    b[11] = S5[temp[3][3]];
    b[12] = S5[temp[3][0]];
    b[13] = S5[temp[2][1]];
    b[14] = S5[temp[1][2]];
    b[15] = S5[temp[0][3]];
    Xor128((byte*)b, (byte*)b, (byte*)covertedClass->m_expandedKey[0]);
}

__device__ int blockDecrypt(struct Rijndael* covertedClass, const byte* input, int inputLen, byte* outBuffer)
{
    if (input == 0 || inputLen <= 0)
        return 0;

    byte block[16], iv[4][4];
    memcpy(iv, covertedClass->m_initVector, 16);


    int numBlocks = inputLen / 16;
    for (int i = numBlocks; i > 0; i--)
    {
        decrypt(covertedClass, input, block);
        Xor128(block, block, (byte*)iv);
#if STRICT_ALIGN
        memcpy(iv, input, 16);
        memcpy(outBuf, block, 16);
#else
        Copy128((byte*)iv, input);
        Copy128(outBuffer, block);
#endif

        input += 16;
        outBuffer += 16;
    }

    memcpy(covertedClass->m_initVector, iv, 16);

    return 16 * numBlocks;
}

////// sha1


/* #define SHA1HANDSOFF * Copies data before messing with it. */

#define rol(value, bits) (((value) << (bits)) | ((value) >> (32 - (bits))))

/* blk0() and blk() perform the initial expand. */
/* I got the idea of expanding during the round function from SSLeay */
#ifdef LITTLE_ENDIAN
#define blk0(i) (block->l[i] = (rol(block->l[i],24)&0xFF00FF00) \
    |(rol(block->l[i],8)&0x00FF00FF))
#else
#define blk0(i) block->l[i]
#endif
#define blk(i) (block->l[i&15] = rol(block->l[(i+13)&15]^block->l[(i+8)&15] \
    ^block->l[(i+2)&15]^block->l[i&15],1))

/* (R0+R1), R2, R3, R4 are the different operations used in SHA1 */
#define R0(v,w,x,y,z,i) {z+=((w&(x^y))^y)+blk0(i)+0x5A827999+rol(v,5);w=rol(w,30);}
#define R1(v,w,x,y,z,i) {z+=((w&(x^y))^y)+blk(i)+0x5A827999+rol(v,5);w=rol(w,30);}
#define R2(v,w,x,y,z,i) {z+=(w^x^y)+blk(i)+0x6ED9EBA1+rol(v,5);w=rol(w,30);}
#define R3(v,w,x,y,z,i) {z+=(((w|x)&y)|(w&x))+blk(i)+0x8F1BBCDC+rol(v,5);w=rol(w,30);}
#define R4(v,w,x,y,z,i) {z+=(w^x^y)+blk(i)+0xCA62C1D6+rol(v,5);w=rol(w,30);}


/* Hash a single 512-bit block. This is the core of the algorithm. */

__device__ void SHA1Transform(uint32 state[5], unsigned char buffer[64])
{
    uint32 a, b, c, d, e;
    typedef union {
        unsigned char c[64];
        uint32 l[16];
    } CHAR64LONG16;
    CHAR64LONG16* block;
#ifdef SHA1HANDSOFF
    static unsigned char workspace[64];
    block = (CHAR64LONG16*)workspace;
    memcpy(block, buffer, 64);
#else
    block = (CHAR64LONG16*)buffer;
#endif
#ifdef SFX_MODULE
    static int pos[80][5];
    static bool pinit = false;
    if (!pinit)
    {
        for (int I = 0, P = 0; I < 80; I++, P = (P ? P - 1 : 4))
        {
            pos[I][0] = P;
            pos[I][1] = (P + 1) % 5;
            pos[I][2] = (P + 2) % 5;
            pos[I][3] = (P + 3) % 5;
            pos[I][4] = (P + 4) % 5;
        }
        pinit = true;
    }
    uint32 s[5];
    for (int I = 0; I < sizeof(s) / sizeof(s[0]); I++)
        s[I] = state[I];

    for (int I = 0; I < 16; I++)
        R0(s[pos[I][0]], s[pos[I][1]], s[pos[I][2]], s[pos[I][3]], s[pos[I][4]], I);
    for (int I = 16; I < 20; I++)
        R1(s[pos[I][0]], s[pos[I][1]], s[pos[I][2]], s[pos[I][3]], s[pos[I][4]], I);
    for (int I = 20; I < 40; I++)
        R2(s[pos[I][0]], s[pos[I][1]], s[pos[I][2]], s[pos[I][3]], s[pos[I][4]], I);
    for (int I = 40; I < 60; I++)
        R3(s[pos[I][0]], s[pos[I][1]], s[pos[I][2]], s[pos[I][3]], s[pos[I][4]], I);
    for (int I = 60; I < 80; I++)
        R4(s[pos[I][0]], s[pos[I][1]], s[pos[I][2]], s[pos[I][3]], s[pos[I][4]], I);

    for (int I = 0; I < sizeof(s) / sizeof(s[0]); I++)
        state[I] += s[I];
#else
    /* Copy context->state[] to working vars */
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    /* 4 rounds of 20 operations each. Loop unrolled. */
    R0(a, b, c, d, e, 0); R0(e, a, b, c, d, 1); R0(d, e, a, b, c, 2); R0(c, d, e, a, b, 3);
    R0(b, c, d, e, a, 4); R0(a, b, c, d, e, 5); R0(e, a, b, c, d, 6); R0(d, e, a, b, c, 7);
    R0(c, d, e, a, b, 8); R0(b, c, d, e, a, 9); R0(a, b, c, d, e, 10); R0(e, a, b, c, d, 11);
    R0(d, e, a, b, c, 12); R0(c, d, e, a, b, 13); R0(b, c, d, e, a, 14); R0(a, b, c, d, e, 15);
    R1(e, a, b, c, d, 16); R1(d, e, a, b, c, 17); R1(c, d, e, a, b, 18); R1(b, c, d, e, a, 19);
    R2(a, b, c, d, e, 20); R2(e, a, b, c, d, 21); R2(d, e, a, b, c, 22); R2(c, d, e, a, b, 23);
    R2(b, c, d, e, a, 24); R2(a, b, c, d, e, 25); R2(e, a, b, c, d, 26); R2(d, e, a, b, c, 27);
    R2(c, d, e, a, b, 28); R2(b, c, d, e, a, 29); R2(a, b, c, d, e, 30); R2(e, a, b, c, d, 31);
    R2(d, e, a, b, c, 32); R2(c, d, e, a, b, 33); R2(b, c, d, e, a, 34); R2(a, b, c, d, e, 35);
    R2(e, a, b, c, d, 36); R2(d, e, a, b, c, 37); R2(c, d, e, a, b, 38); R2(b, c, d, e, a, 39);
    R3(a, b, c, d, e, 40); R3(e, a, b, c, d, 41); R3(d, e, a, b, c, 42); R3(c, d, e, a, b, 43);
    R3(b, c, d, e, a, 44); R3(a, b, c, d, e, 45); R3(e, a, b, c, d, 46); R3(d, e, a, b, c, 47);
    R3(c, d, e, a, b, 48); R3(b, c, d, e, a, 49); R3(a, b, c, d, e, 50); R3(e, a, b, c, d, 51);
    R3(d, e, a, b, c, 52); R3(c, d, e, a, b, 53); R3(b, c, d, e, a, 54); R3(a, b, c, d, e, 55);
    R3(e, a, b, c, d, 56); R3(d, e, a, b, c, 57); R3(c, d, e, a, b, 58); R3(b, c, d, e, a, 59);
    R4(a, b, c, d, e, 60); R4(e, a, b, c, d, 61); R4(d, e, a, b, c, 62); R4(c, d, e, a, b, 63);
    R4(b, c, d, e, a, 64); R4(a, b, c, d, e, 65); R4(e, a, b, c, d, 66); R4(d, e, a, b, c, 67);
    R4(c, d, e, a, b, 68); R4(b, c, d, e, a, 69); R4(a, b, c, d, e, 70); R4(e, a, b, c, d, 71);
    R4(d, e, a, b, c, 72); R4(c, d, e, a, b, 73); R4(b, c, d, e, a, 74); R4(a, b, c, d, e, 75);
    R4(e, a, b, c, d, 76); R4(d, e, a, b, c, 77); R4(c, d, e, a, b, 78); R4(b, c, d, e, a, 79);
    /* Add the working vars back into context.state[] */
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;

    /* Wipe variables */
    a = b = c = d = e = 0;
    memset(&a, 0, sizeof(a));
#endif
}


/* Initialize new context */

__device__ void hash_initial(hash_context* context)
{
    /* SHA1 initialization constants */
    context->state[0] = 0x67452301;
    context->state[1] = 0xEFCDAB89;
    context->state[2] = 0x98BADCFE;
    context->state[3] = 0x10325476;
    context->state[4] = 0xC3D2E1F0;
    context->count[0] = context->count[1] = 0;
}


/* Run your data through this. */
__device__ void hash_process(hash_context* context, unsigned char* data, unsigned len)
{
    unsigned int i, j;
    uint blen = ((uint)len) << 3;

    j = (context->count[0] >> 3) & 63;
    if ((context->count[0] += blen) < blen) context->count[1]++;
    context->count[1] += (len >> 29);
    if ((j + len) > 63) {
        memcpy(&context->buffer[j], data, (i = 64 - j));
        SHA1Transform(context->state, context->buffer);
        for (; i + 63 < len; i += 64) {
#ifdef ALLOW_NOT_ALIGNED_INT
            SHA1Transform(context->state, &data[i]);
#else
            unsigned char buffer[64];
            memcpy(buffer, data + i, sizeof(buffer));
            SHA1Transform(context->state, buffer);
            memcpy(data + i, buffer, sizeof(buffer));
#endif
#ifdef BIG_ENDIAN
            unsigned char* d = data + i;
            for (int k = 0; k < 64; k += 4)
            {
                byte b0 = d[k], b1 = d[k + 1];
                d[k] = d[k + 3];
                d[k + 1] = d[k + 2];
                d[k + 2] = b1;
                d[k + 3] = b0;
            }
#endif
        }
        j = 0;
    }
    else i = 0;
    if (len > i)
        memcpy(&context->buffer[j], &data[i], len - i);
}


/* Add padding and return the message digest. */

__device__ void hash_final(hash_context* context, uint32 digest[5])
{
    uint i, j;
    unsigned char finalcount[8];

    for (i = 0; i < 8; i++) {
        finalcount[i] = (unsigned char)((context->count[(i >= 4 ? 0 : 1)]
            >> ((3 - (i & 3)) * 8)) & 255);  /* Endian independent */
    }
    unsigned char ch = '\200';
    hash_process(context, &ch, 1);
    while ((context->count[0] & 504) != 448) {
        ch = 0;
        hash_process(context, &ch, 1);
    }
    hash_process(context, finalcount, 8);  /* Should cause a SHA1Transform() */
    for (i = 0; i < 5; i++) {
        digest[i] = context->state[i] & 0xffffffff;
    }
    /* Wipe variables */
    memset(&i, 0, sizeof(i));
    memset(&j, 0, sizeof(j));
    memset(context->buffer, 0, 64);
    memset(context->state, 0, 20);
    memset(context->count, 0, 8);
    memset(&finalcount, 0, 8);
#ifdef SHA1HANDSOFF  /* make SHA1Transform overwrite it's own static vars */
    SHA1Transform(context->state, context->buffer);
#endif
}


__device__ int strlenGPU(const char* buf)
{
	for(int i = 0 ; i < 50; i++)
	{
		if(buf[i] == '\0')
		{
            return i;
		}
	}
}


__device__ char *ucs2_str(char *dst, const char *src)
{
    char *ret;

    ret = dst;
    for (int i = 0 ; i < 5 ; i++)
    {
        *dst++ = *src++;
        *dst++ = '\0';
    }
    return ret;
}

__device__ int ucs2_len(const char *s)
{
    //return 2 * strlenGPU(s);
    return 2 * 5;
}

#define PAT_ANYCHAR '?'
#define PAT_CHARS "0123456789" \
                  "abcdefghijklmnopqrstuvwxyz"

/*
** Increment the char pointed to by 'c', according to PAT_CHARS.
**
** Return 1 if the increment has generated a carry, else return 0.
*/
int inc_chr(char *c)
{
    if (*c >= '0' && *c < '9')
        (*c)++;
    else if (*c == '9')
        (*c) = 'a';
    else if (*c >= 'a' && *c < 'z')
        (*c)++;
    else if (*c == 'z')
    {
        (*c) = '0';
        return 1;
    }
    else
        fprintf(stderr, "Bug: c == '%c'\n", *c), exit(1);
    return 0;
}

/*
** Borrowed from sources of unrar
*/
__device__ void gen_aes_val(unsigned char *aes_key, unsigned char *aes_iv,
                 char *pwd, int plen, unsigned char *salt)
{
    unsigned char *RawPsw = (unsigned char *)malloc(plen + 8);
    int RawLength = plen + 8;
    hash_context c;

    memcpy(RawPsw, pwd, plen);
    memcpy(RawPsw + plen, salt, 8);

    hash_initial(&c);
    const int HashRounds = 0x40000;

    for (int I = 0; I < HashRounds; I++)
    {
        hash_process(&c, RawPsw, RawLength);
        byte PswNum[3];
        PswNum[0] = (byte)I;
        PswNum[1] = (byte)(I >> 8);
        PswNum[2] = (byte)(I >> 16);
        hash_process(&c, PswNum, 3);
        if (I % (HashRounds / 16) == 0)
        {
            hash_context tempc = c;
            uint32 digest[5];
            hash_final(&tempc, digest);
            aes_iv[I / (HashRounds / 16)] = (byte)digest[4];
        }
    }
	// free RawPsw
    free(RawPsw);
	//
    uint32 digest[5];
    hash_final(&c, digest);
    for (int I = 0; I < 4; I++)
        for (int J = 0; J < 4; J++)
            aes_key[I * 4 + J] = (byte)(digest[I] >> (J * 8));
}

/*
** Test a password, using the end-of-archive block.
**
**   @salt point to 8 bytes of salt data
**   @in is a 16-byte encrypted block representing the end-of-archive block
**   @p is a password to try
**
** Return true on good passwd, false on bad passwd.
*/


__device__ int memcmpGPU(unsigned char* buf1 , const char* buf2 , int size)
{
	for(int i = 0 ; i < size ; i++)
	{
        if (buf1[i] != (unsigned char)buf2[i])
        {
            return 1;
        }
	}
    return 0;
}


__device__ int test_password(unsigned char *salt, unsigned char *in, const char *p)
{
    unsigned char aes_key[16];
    unsigned char aes_iv[16];
    struct Rijndael rin;
    unsigned char out[16];
    char p2[MAXPASSWD];
    Direction diretion = Decrypt;
    RijndaelInit(&rin);
    
    gen_aes_val(aes_key, aes_iv, ucs2_str(p2, p), ucs2_len(p), salt);
    
    init(&rin, diretion, aes_key, aes_iv);
    blockDecrypt(&rin, in, 16, out);
    
    ///* cmp with the usual end-of-archive block present in (all?) rar files */

    return !memcmpGPU(out, "\xc4\x3d\x7b\x00\x40\x07\x00", 7);
}

int readFile(const char *f, unsigned char *buf)
{
    int fd;
    int n;

    if (-1 == (fd = _open(f, O_RDONLY, 0)))
        return (1);
    if (-1 == _lseek(fd, -24, SEEK_END))
        return (2);
    n = _read(fd, buf, 24);
    if (!n)
        return (2);
    if (n != 24)
        return (2);
    _close(fd);
    return (0);
}

// this function test password
__device__ void doTaskGPU(char* pass, int* isFinish, unsigned char* buf)
{
    int result = 0;
    result = test_password(buf, buf + 8, pass);
    if (result == 1)
    {
        printf("password %s passed\n", pass);
    }
    else
    {
        printf("password %s failed\n", pass);
    }
       //*isFinish = 1;
}

// try to gen password from id of thread
__device__ char* genPass(char* pattern, int id , const char* PAT_CHARS_GPU)
{
	// allocate new device memory of result and result char
    int* result =(int*)malloc(6 * sizeof(int));
    char* resultChar = (char*)malloc(6 * sizeof(char));
    
	// assign value from pattern to resultChar
	for(int i = 0; i < 5 ; i++)
	{
        resultChar[i] = pattern[i];
	}
    
	// add end of string value
    resultChar[5] = '\0';
	//
    //int lengthPattern = strlenGPU(pattern);
    //
    //
    int numberOfAnyChar = 0;
	for(int i = 0 ; i < 5 ; i++)
	{
		if(pattern[i] == '?')
		{
            numberOfAnyChar++;
		}
	}
    
    int j = id;
	for(int i = numberOfAnyChar - 1 ; i >= 0 ; i--)
	{
		if((j / powf(36,i)) >= 36)
		{
            return nullptr;
		}
        *result = j / powf(36, i);
        //printf("j=%d result=%d  j/(36^i)=%d 36^i=%d     ", j ,*result, j / powf(36, i), powf(36, i));
        j = j - *result * powf(36, i);
        result++;
	}
    
    // last index of pass
    
    for (int i = 0 ; i < numberOfAnyChar ; i++)
    {
        for (int j = 5 - 1; j >= 0; j--)
        {
            if (resultChar[j] == '?')
            {
                resultChar[j] = PAT_CHARS_GPU[*(result -1)];
                //printf("result is %d", *result);
                result--;
                break;
            }
        }
    }
    //printf("password is generete is :");
	/*for(int i = 0 ; i < 5 ; i++)
	{
        printf("%c", resultChar[i]);
	}*/

	// free memory
    free(result);
    return resultChar;
}
// kernel function to test password with shared mem( isFinish , pattern and buf ) index variable is copied from host to make enough id
__global__ void gpuDoWork(int* isFinish , char* pattern ,unsigned char* buf , char* PAT_CHARS_GPU , int* index)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x + *index;
    
   char* passPrivate = genPass(pattern, id, PAT_CHARS_GPU);
	if(passPrivate != nullptr)
	{
        doTaskGPU(passPrivate, isFinish, buf);
        free(passPrivate);
	}
    else
    {
        printf("pass is not in range");
        free(passPrivate);
    }
    //printf("%s", passPrivate);
}


int main()
{
	// path to rar3 file
    const char *file = "C:\\ProgramData\\NVIDIA Corporation\\CUDA Samples\\v11.2\\bin\\win64\\Debug\\unrar.rar";
	// pattern of password : ? is any
    const char *pattern = "123??";
    unsigned char *buf = (unsigned char *)malloc(24 * sizeof(unsigned char));
    int re = readFile(file, buf);
    if (re == 0)
    {
        printf("read file success");
        int numberOfTasks = 1; //
        for (int i = 0 ; i < strlen(pattern); i++)
        {
            if (pattern[i] == '?')
            {
                numberOfTasks *= 36;
            }
        }
        printf("There're %d tasks need to be executed ", numberOfTasks);

    	// set device
        checkCudaErrors(cudaSetDevice(0));

    	// specify number of block and thread to run
        const int ThreadPerBlock = 512;
        const int MaxBlocks = 10;

    	// shared memory pointer in host
    	// isFinish to check if result is found
    	// PAT is character
    	// 
        int* isFinish;
        char* patternGPU;
        char* PAT;
        unsigned char* bufGPU;
    	
    	// Allocate shared memory in device
        cudaMallocManaged(&isFinish, sizeof(int));
        cudaMallocManaged(&patternGPU, 5 * sizeof(char));
        cudaMallocManaged(&PAT, 36 * sizeof(char));
        int sizeCopy = 24;
        cudaMallocManaged(&bufGPU, sizeCopy * sizeof(unsigned char));
        
        
    	
    	// Set value of shared memory in host
        *isFinish = 0;
        for (int i = 0; i < 5; i++)
        {
            patternGPU[i] = pattern[i];
        }
        for (int i = 0; i < 36; i++)
        {
            PAT[i] = PAT_CHARS[i];
        }
    	for(int i = 0 ; i < sizeCopy ; i++ )
    	{
            bufGPU[i] = buf[i];
    	}
    	//

    	// declare variables in host
        int indexSource = 0;
    	// 
        
        omp_set_num_threads(10);
//#pragma omp parallel for schedule(static) default(none)
        	for(int i = 0 ; i < 10 ; i++)
        	{
        		// set index from host
                indexSource = (ThreadPerBlock * MaxBlocks) * i;
                printf("\n");
                dim3 threadBlock = ThreadPerBlock;
                dim3 block = MaxBlocks;
        		// declare index
                int* index = (int*)malloc(sizeof(int*));
                checkCudaErrors(cudaMalloc((void**)&index, sizeof(int)));
                
        		// copy memory from host to device
                checkCudaErrors(cudaMemcpy(index, &indexSource, sizeof(int), cudaMemcpyHostToDevice));

        		
                gpuDoWork <<< block, threadBlock >>> (isFinish, patternGPU, bufGPU, PAT , index);
                cudaDeviceSynchronize();
                // free index
                cudaFree(index);
        	}
        return 1; /* if we reach this point, passwd has not been found */
    }
    else
    {
        printf("failed");
    }
}