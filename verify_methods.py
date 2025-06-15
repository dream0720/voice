"""
快速验证 modern_voice_app.py 的所有核心方法
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def check_methods():
    """检查所有核心方法是否存在"""
    try:
        # 导入主类
        from gui.modern_voice_app import ModernVoiceProcessingApp
        
        # 要检查的核心方法列表
        required_methods = [
            'process_preprocessing',
            'process_source_separation', 
            'process_speaker_separation',
            'process_voice_matching',
            'view_preprocessing_results',
            'view_source_separation_results',
            'view_speaker_separation_results',
            'view_voice_matching_results',
            'open_output_folder',
            'get_current_card',
            'update_progress',
            'handle_processing_complete',
            'handle_processing_error'
        ]
        
        print("🔍 检查 ModernVoiceProcessingApp 的核心方法...")
        
        missing_methods = []
        existing_methods = []
        
        for method_name in required_methods:
            if hasattr(ModernVoiceProcessingApp, method_name):
                existing_methods.append(method_name)
                print(f"  ✅ {method_name}")
            else:
                missing_methods.append(method_name)
                print(f"  ❌ {method_name} - 缺失!")
        
        print(f"\n📊 总结:")
        print(f"  ✅ 存在的方法: {len(existing_methods)}")
        print(f"  ❌ 缺失的方法: {len(missing_methods)}")
        
        if missing_methods:
            print(f"\n⚠️  缺失的方法列表:")
            for method in missing_methods:
                print(f"    - {method}")
            return False
        else:
            print(f"\n🎉 所有核心方法都存在!")
            return True
            
    except Exception as e:
        print(f"❌ 检查失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_methods()
    if success:
        print("\n✅ 方法检查通过!")
    else:
        print("\n❌ 方法检查失败!")
    
    input("\n按回车键退出...")
