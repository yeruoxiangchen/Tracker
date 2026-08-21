Shader "Tracker/ARMeshScreenSpaceOutline"
{
    Properties
    {
        [PerRendererData] _MainTex ("Silhouette Mask", 2D) = "black" {}
        _OutlineColor ("Outline Color", Color) = (0.05, 1.0, 0.70, 1.0)
        _OutlineWidthPixels ("Outline Width In Mask Pixels", Range(1, 6)) = 2
    }

    SubShader
    {
        Tags
        {
            "Queue"="Overlay"
            "RenderType"="Transparent"
            "IgnoreProjector"="True"
        }
        Cull Off
        Lighting Off
        ZWrite Off
        ZTest Always
        Blend SrcAlpha OneMinusSrcAlpha

        Pass
        {
            Name "OutsideBoundary"

            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            sampler2D _MainTex;
            float4 _MainTex_ST;
            float4 _MainTex_TexelSize;
            fixed4 _OutlineColor;
            float _OutlineWidthPixels;

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float4 vertex : SV_POSITION;
                float2 uv : TEXCOORD0;
            };

            v2f vert(appdata input)
            {
                v2f output;
                output.vertex = UnityObjectToClipPos(input.vertex);
                output.uv = TRANSFORM_TEX(input.uv, _MainTex);
                return output;
            }

            fixed4 frag(v2f input) : SV_Target
            {
                float center = tex2D(_MainTex, input.uv).a;
                float2 delta = _MainTex_TexelSize.xy
                    * clamp(_OutlineWidthPixels, 1.0, 6.0);
                float2 diagonal = delta * 0.70710678;
                float neighbor = 0.0;
                neighbor = max(neighbor, tex2D(_MainTex, input.uv + float2(delta.x, 0)).a);
                neighbor = max(neighbor, tex2D(_MainTex, input.uv - float2(delta.x, 0)).a);
                neighbor = max(neighbor, tex2D(_MainTex, input.uv + float2(0, delta.y)).a);
                neighbor = max(neighbor, tex2D(_MainTex, input.uv - float2(0, delta.y)).a);
                neighbor = max(neighbor, tex2D(_MainTex, input.uv + diagonal).a);
                neighbor = max(neighbor, tex2D(_MainTex, input.uv - diagonal).a);
                neighbor = max(neighbor, tex2D(_MainTex, input.uv + float2(diagonal.x, -diagonal.y)).a);
                neighbor = max(neighbor, tex2D(_MainTex, input.uv + float2(-diagonal.x, diagonal.y)).a);

                // Match the server audit: dilated silhouette XOR original
                // silhouette, so only the outside cyan boundary is visible.
                float outsideBoundary = saturate(neighbor - center);
                return fixed4(
                    _OutlineColor.rgb,
                    _OutlineColor.a * outsideBoundary
                );
            }
            ENDCG
        }
    }
}
